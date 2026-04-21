/*
 * Baseline naive CUDA kernels for the auto-tuner.
 * RTX 2070 (sm_75, Turing). These are intentionally unoptimized —
 * they serve as the reference point from which the auto-tuner improves.
 *
 * Kernels:
 *   1. Matrix multiplication (global memory only)
 *   2. Softmax (naive, uncoalesced)
 *   3. Parallel reduction (naive, bank conflicts)
 *   4. Layer normalization (naive, two-pass)
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* ─────────────────────────────────────────────────────────────────────────
 * 1. MATRIX MULTIPLICATION — naive O(N³), one thread per output element
 *    Memory pattern: strided reads of B (column-wise) → poor coalescing
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void matmul_naive(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];  // B read is non-coalesced
        C[row * N + col] = sum;
    }
}

/* ─────────────────────────────────────────────────────────────────────────
 * 2. SOFTMAX — naive, each block handles one row
 *    Issues: two passes over global memory, no warp-level reduction
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void softmax_naive(const float* __restrict__ input,
                              float* __restrict__ output,
                              int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row  = input  + row * cols;
    float*       out_row = output + row * cols;

    // Pass 1: find max (for numerical stability)
    float max_val = in_row[0];
    for (int i = 1; i < cols; ++i)
        max_val = fmaxf(max_val, in_row[i]);

    // Pass 2: compute exp and sum
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) {
        out_row[i] = expf(in_row[i] - max_val);
        sum += out_row[i];
    }

    // Pass 3: normalize
    for (int i = 0; i < cols; ++i)
        out_row[i] /= sum;
}

/* ─────────────────────────────────────────────────────────────────────────
 * 3. PARALLEL REDUCTION — naive, sequential addressing
 *    Issues: divergent warps in early steps, bank conflicts, idle threads
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void reduction_naive(const float* __restrict__ input,
                                float* __restrict__ output,
                                int N)
{
    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (gid < N) ? input[gid] : 0.0f;
    __syncthreads();

    // Sequential addressing — half the threads idle at each step
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0)          // divergent warp execution
            sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = sdata[0];
}

/* ─────────────────────────────────────────────────────────────────────────
 * 4. LAYER NORMALIZATION — naive two-pass
 *    Issues: reads full vector twice from global memory, no shared memory
 * ───────────────────────────────────────────────────────────────────────── */
__global__ void layernorm_naive(const float* __restrict__ input,
                                float* __restrict__ output,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                int rows, int cols,
                                float eps)
{
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row  = input  + row * cols;
    float*       out_row = output + row * cols;

    // Pass 1: mean
    float mean = 0.0f;
    for (int i = 0; i < cols; ++i)
        mean += in_row[i];
    mean /= cols;

    // Pass 2: variance
    float var = 0.0f;
    for (int i = 0; i < cols; ++i) {
        float diff = in_row[i] - mean;
        var += diff * diff;
    }
    var /= cols;

    float inv_std = rsqrtf(var + eps);

    // Pass 3: normalize + scale + shift
    for (int i = 0; i < cols; ++i)
        out_row[i] = gamma[i] * (in_row[i] - mean) * inv_std + beta[i];
}

/* ─────────────────────────────────────────────────────────────────────────
 * Host-side launcher helpers (called by benchmark harness)
 * ───────────────────────────────────────────────────────────────────────── */

// Returns elapsed milliseconds
float launch_matmul(float* d_A, float* d_B, float* d_C, int N,
                    int block_dim, int warmup, int iters)
{
    dim3 block(block_dim, block_dim);
    dim3 grid((N + block_dim - 1) / block_dim,
              (N + block_dim - 1) / block_dim);

    // Warmup
    for (int i = 0; i < warmup; ++i)
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iters; ++i)
        matmul_naive<<<grid, block>>>(d_A, d_B, d_C, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

float launch_softmax(float* d_in, float* d_out, int rows, int cols,
                     int warmup, int iters)
{
    dim3 grid(rows);
    dim3 block(1);  // naive: single-threaded per row

    for (int i = 0; i < warmup; ++i)
        softmax_naive<<<grid, block>>>(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iters; ++i)
        softmax_naive<<<grid, block>>>(d_in, d_out, rows, cols);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

float launch_reduction(float* d_in, float* d_out, int N,
                       int block_size, int warmup, int iters)
{
    int grid_size = (N + block_size - 1) / block_size;
    size_t smem = block_size * sizeof(float);

    for (int i = 0; i < warmup; ++i)
        reduction_naive<<<grid_size, block_size, smem>>>(d_in, d_out, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iters; ++i)
        reduction_naive<<<grid_size, block_size, smem>>>(d_in, d_out, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}

float launch_layernorm(float* d_in, float* d_out, float* d_gamma,
                       float* d_beta, int rows, int cols,
                       int warmup, int iters)
{
    dim3 grid(rows);
    dim3 block(1);

    for (int i = 0; i < warmup; ++i)
        layernorm_naive<<<grid, block>>>(d_in, d_out, d_gamma, d_beta,
                                        rows, cols, 1e-5f);
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    for (int i = 0; i < iters; ++i)
        layernorm_naive<<<grid, block>>>(d_in, d_out, d_gamma, d_beta,
                                        rows, cols, 1e-5f);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms / iters;
}
