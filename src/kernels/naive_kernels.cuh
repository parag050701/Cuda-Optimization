/*
 * naive_kernels.cuh — reference (naive) implementations of all four kernels.
 *
 * These are intentionally unoptimized. They serve two roles:
 *   1. Baseline timing reference (benchmark_runner.cu).
 *   2. Correctness reference inside each auto-tuned variant binary
 *      (generator.py embeds an #include of this header and runs the naive
 *      kernel on the same inputs, then compares element-wise).
 *
 * Header-only + static inline where appropriate so multiple TUs can include
 * it without ODR issues.
 */

#ifndef CUDA_AUTOTUNER_NAIVE_KERNELS_CUH
#define CUDA_AUTOTUNER_NAIVE_KERNELS_CUH

#include <cuda_runtime.h>
#include <math.h>

/* ── Kernels ────────────────────────────────────────────────────────────── */

static __global__ void matmul_naive(const float* __restrict__ A,
                                    const float* __restrict__ B,
                                    float* __restrict__ C, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float s = 0.0f;
        for (int k = 0; k < N; ++k) s += A[row * N + k] * B[k * N + col];
        C[row * N + col] = s;
    }
}

static __global__ void softmax_naive(const float* __restrict__ in,
                                     float* __restrict__ out,
                                     int rows, int cols)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* ir  = in  + row * cols;
    float*       or_ = out + row * cols;
    float mx = ir[0];
    for (int i = 1; i < cols; ++i) mx = fmaxf(mx, ir[i]);
    float s = 0.0f;
    for (int i = 0; i < cols; ++i) { or_[i] = expf(ir[i] - mx); s += or_[i]; }
    for (int i = 0; i < cols; ++i) or_[i] /= s;
}

static __global__ void reduction_naive(const float* __restrict__ in,
                                       float* __restrict__ out, int N)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    sdata[tid] = (gid < N) ? in[gid] : 0.0f;
    __syncthreads();
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = sdata[0];
}

static __global__ void layernorm_naive(const float* __restrict__ in,
                                       float* __restrict__ out,
                                       const float* __restrict__ gamma,
                                       const float* __restrict__ beta,
                                       int rows, int cols, float eps)
{
    int row = blockIdx.x;
    if (row >= rows) return;
    const float* ir  = in  + row * cols;
    float*       or_ = out + row * cols;
    float mean = 0.0f;
    for (int i = 0; i < cols; ++i) mean += ir[i];
    mean /= cols;
    float var = 0.0f;
    for (int i = 0; i < cols; ++i) { float d = ir[i] - mean; var += d * d; }
    var /= cols;
    float inv_std = rsqrtf(var + eps);
    for (int i = 0; i < cols; ++i)
        or_[i] = gamma[i] * (ir[i] - mean) * inv_std + beta[i];
}

/* ── Host-side launchers (match the geometry of the baseline runner) ────── */

static inline void launch_matmul_naive(const float* A, const float* B,
                                       float* C, int N, int blk)
{
    dim3 block(blk, blk);
    dim3 grid((N + blk - 1) / blk, (N + blk - 1) / blk);
    matmul_naive<<<grid, block>>>(A, B, C, N);
}

static inline void launch_softmax_naive(const float* in, float* out,
                                        int rows, int cols)
{
    softmax_naive<<<rows, 1>>>(in, out, rows, cols);
}

static inline void launch_reduction_naive(const float* in, float* out,
                                          int N, int blk)
{
    int grid = (N + blk - 1) / blk;
    reduction_naive<<<grid, blk, blk * sizeof(float)>>>(in, out, N);
}

static inline void launch_layernorm_naive(const float* in, float* out,
                                          const float* gamma, const float* beta,
                                          int rows, int cols)
{
    layernorm_naive<<<rows, 1>>>(in, out, gamma, beta, rows, cols, 1e-5f);
}

#endif  /* CUDA_AUTOTUNER_NAIVE_KERNELS_CUH */
