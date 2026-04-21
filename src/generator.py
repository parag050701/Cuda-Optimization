"""
generator.py — template-based CUDA code generator.

Takes a kernel profile + a concrete parameter configuration and emits
a .cu file. The generated file includes the optimized kernel and a
benchmark_runner main() so it can be compiled and timed standalone.
"""

import itertools
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from parser import KernelProfile, build_search_space

ROOT        = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
GEN_DIR     = RESULTS_DIR / "generated"
GEN_DIR.mkdir(parents=True, exist_ok=True)

ARCH = "sm_75"  # RTX 2070 (Turing)


# ── Kernel templates ────────────────────────────────────────────────────────

MATMUL_TEMPLATE = """\
/*
 * Generated matmul variant
 * block={block_size}x{block_size}  tile={tile_x}x{tile_y}  unroll={unroll}
 * transpose_B={transpose_b}
 */
#include <cuda_runtime.h>

#define TILE_X {tile_x}
#define TILE_Y {tile_y}
#define BLOCK  {block_size}

__global__ void matmul_opt(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C, int N)
{{
    __shared__ float As[TILE_Y][TILE_X];
    __shared__ float Bs[TILE_Y][TILE_X];

    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * TILE_Y + ty;
    int col = blockIdx.x * TILE_X + tx;
    float sum = 0.0f;

    for (int t = 0; t < (N + TILE_X - 1) / TILE_X; ++t) {{
        // Load tile of A (row-major, coalesced)
        if (row < N && t * TILE_X + tx < N)
            As[ty][tx] = A[row * N + t * TILE_X + tx];
        else
            As[ty][tx] = 0.0f;

        // Load tile of B (optionally transposed for coalescing)
{b_load}
        __syncthreads();

        #pragma unroll {unroll}
        for (int k = 0; k < TILE_X; ++k)
            sum += As[ty][k] * Bs[k][tx];

        __syncthreads();
    }}

    if (row < N && col < N)
        C[row * N + col] = sum;
}}
"""

REDUCTION_TEMPLATE = """\
/*
 * Generated reduction variant
 * block={block_size}  unroll={unroll}  warp_shuffle={warp_shuffle}
 */
#include <cuda_runtime.h>

#define BLOCK {block_size}

__global__ void reduction_opt(const float* __restrict__ input,
                              float* __restrict__ output, int N)
{{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * (blockDim.x * 2) + tid;

    // Load two elements per thread (sequential addressing, no bank conflicts)
    float val = 0.0f;
    if (gid < N)             val  = input[gid];
    if (gid + blockDim.x < N) val += input[gid + blockDim.x];
    sdata[tid] = val;
    __syncthreads();

{reduce_body}

    if (tid == 0) output[blockIdx.x] = sdata[0];
}}
"""

SOFTMAX_TEMPLATE = """\
/*
 * Generated softmax variant
 * block={block_size}  unroll={unroll}
 */
#include <cuda_runtime.h>

#define BLOCK {block_size}

__global__ void softmax_opt(const float* __restrict__ input,
                            float* __restrict__ output,
                            int rows, int cols)
{{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    if (row >= rows) return;

    const float* in_row  = input  + row * cols;
    float*       out_row = output + row * cols;

    // Phase 1: parallel max reduction into smem
    float thread_max = -1e38f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        thread_max = fmaxf(thread_max, in_row[i]);
    smem[threadIdx.x] = thread_max;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x+s]);
        __syncthreads();
    }}
    float row_max = smem[0];
    __syncthreads();

    // Phase 2: exp + partial sum
    float thread_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {{
        float e = expf(in_row[i] - row_max);
        out_row[i] = e;
        thread_sum += e;
    }}
    smem[threadIdx.x] = thread_sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {{
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }}
    float total = smem[0];

    // Phase 3: normalize
    #pragma unroll {unroll}
    for (int i = threadIdx.x; i < cols; i += blockDim.x)
        out_row[i] /= total;
}}
"""

BENCHMARK_MAIN = """\

/* ── Benchmark driver ────────────────────────────────────────────────── */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#define CUDA_CHECK(call) \\
    do {{ cudaError_t e=(call); if(e!=cudaSuccess){{ \\
        fprintf(stderr,"CUDA %s:%d %s\\n",__FILE__,__LINE__,cudaGetErrorString(e)); \\
        exit(1); }} }} while(0)

static void fill_random(float* d, int n) {{
    float* h=(float*)malloc(n*sizeof(float));
    for(int i=0;i<n;i++) h[i]=(float)rand()/RAND_MAX;
    CUDA_CHECK(cudaMemcpy(d,h,n*sizeof(float),cudaMemcpyHostToDevice));
    free(h);
}}

int main(void) {{
    int warmup = getenv("WARMUP") ? atoi(getenv("WARMUP")) : 5;
    int iters  = getenv("ITERS")  ? atoi(getenv("ITERS"))  : 100;

    {setup_code}

    // Warmup
    for(int i=0;i<warmup;i++) {{ {kernel_launch} }}
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start,stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float total=0,mn=FLT_MAX,mx=0;
    for(int i=0;i<iters;i++) {{
        CUDA_CHECK(cudaEventRecord(start));
        {kernel_launch}
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms=0; CUDA_CHECK(cudaEventElapsedTime(&ms,start,stop));
        total+=ms; if(ms<mn) mn=ms; if(ms>mx) mx=ms;
    }}
    printf("TIMING {variant_tag} %.4f %.4f %.4f %d\\n",
           total/iters, mn, mx, iters);

    {cleanup_code}
    return 0;
}}
"""


# ── Code generation helpers ─────────────────────────────────────────────────

def _matmul_b_load(transpose_b: bool, tile_x: int, tile_y: int) -> str:
    if transpose_b:
        return (
            "        // Load B transposed: thread reads B col-major → coalesced\n"
            "        if (t * TILE_Y + ty < N && col < N)\n"
            "            Bs[ty][tx] = B[col * N + t * TILE_Y + ty];\n"
            "        else\n"
            "            Bs[ty][tx] = 0.0f;"
        )
    return (
        "        // Load B row-major (original)\n"
        "        if (t * TILE_Y + ty < N && col < N)\n"
        "            Bs[ty][tx] = B[(t * TILE_Y + ty) * N + col];\n"
        "        else\n"
        "            Bs[ty][tx] = 0.0f;"
    )


def _reduction_body(block_size: int, warp_shuffle: bool) -> str:
    lines = []
    if warp_shuffle:
        lines.append(
            "    // Tree reduction in shared memory down to warp boundary\n"
            "    for (int s = blockDim.x / 2; s >= 32; s >>= 1) {\n"
            "        if (tid < s) sdata[tid] += sdata[tid + s];\n"
            "        __syncthreads();\n"
            "    }\n"
            "    // Warp-level reduction with shuffle\n"
            "    if (tid < 32) {\n"
            "        float v = sdata[tid];\n"
            "        v += __shfl_down_sync(0xffffffff, v, 16);\n"
            "        v += __shfl_down_sync(0xffffffff, v,  8);\n"
            "        v += __shfl_down_sync(0xffffffff, v,  4);\n"
            "        v += __shfl_down_sync(0xffffffff, v,  2);\n"
            "        v += __shfl_down_sync(0xffffffff, v,  1);\n"
            "        if (tid == 0) sdata[0] = v;\n"
            "    }"
        )
    else:
        lines.append(
            "    for (int s = blockDim.x / 2; s > 0; s >>= 1) {\n"
            "        if (tid < s) sdata[tid] += sdata[tid + s];\n"
            "        __syncthreads();\n"
            "    }"
        )
    return "\n".join(lines)


def generate_matmul(params: dict) -> str:
    b_load = _matmul_b_load(params["transpose_b"], params["tile_x"], params["tile_y"])
    kernel = MATMUL_TEMPLATE.format(b_load=b_load, **params)

    N      = 1024
    bs     = params["tile_x"]   # tile_x == tile_y for square tiles
    tag    = _variant_tag("matmul", params)
    setup  = (f"int N={N}; float *A,*B,*C;\n"
              f"    cudaMalloc(&A,N*N*4); cudaMalloc(&B,N*N*4); cudaMalloc(&C,N*N*4);\n"
              f"    fill_random(A,N*N); fill_random(B,N*N);")
    launch = (f"matmul_opt<<<dim3((N+{bs}-1)/{bs},(N+{bs}-1)/{bs}),"
              f"dim3({bs},{bs})>>>(A,B,C,N);")
    cleanup = "cudaFree(A); cudaFree(B); cudaFree(C);"

    return kernel + BENCHMARK_MAIN.format(
        variant_tag=tag, setup_code=setup,
        kernel_launch=launch, cleanup_code=cleanup
    )


def generate_reduction(params: dict) -> str:
    body   = _reduction_body(params["block_size"], params["warp_shuffle"])
    kernel = REDUCTION_TEMPLATE.format(reduce_body=body, **params)

    N    = 1 << 20
    blk  = params["block_size"]
    tag  = _variant_tag("reduction", params)
    setup   = (f"int N={N}; int grid=(N+{blk*2}-1)/({blk*2});\n"
               f"    float *in,*out; cudaMalloc(&in,N*4); cudaMalloc(&out,grid*4);\n"
               f"    fill_random(in,N);")
    launch  = f"reduction_opt<<<grid,{blk},{blk}*4>>>(in,out,N);"
    cleanup = "cudaFree(in); cudaFree(out);"

    return kernel + BENCHMARK_MAIN.format(
        variant_tag=tag, setup_code=setup,
        kernel_launch=launch, cleanup_code=cleanup
    )


def generate_softmax(params: dict) -> str:
    kernel = SOFTMAX_TEMPLATE.format(**params)

    rows = 1024; cols = 4096
    blk  = params["block_size"]
    tag  = _variant_tag("softmax", params)
    setup   = (f"int rows={rows},cols={cols}; float *in,*out;\n"
               f"    cudaMalloc(&in,rows*cols*4); cudaMalloc(&out,rows*cols*4);\n"
               f"    fill_random(in,rows*cols);")
    launch  = f"softmax_opt<<<rows,{blk},{blk}*4>>>(in,out,rows,cols);"
    cleanup = "cudaFree(in); cudaFree(out);"

    return kernel + BENCHMARK_MAIN.format(
        variant_tag=tag, setup_code=setup,
        kernel_launch=launch, cleanup_code=cleanup
    )


def _variant_tag(kernel: str, params: dict) -> str:
    parts = [kernel]
    for k, v in sorted(params.items()):
        parts.append(f"{k}{v}")
    return "_".join(str(p) for p in parts)


GENERATORS = {
    "matmul":    generate_matmul,
    "softmax":   generate_softmax,
    "reduction": generate_reduction,
}


def enumerate_variants(kernel: str, space: dict) -> list[tuple[dict, Path]]:
    """
    Expand the search space into a list of (params, output_path) tuples.
    Returns only valid combinations (e.g., tile_x == tile_y for matmul).
    """
    param_keys = [k for k in space if not k.startswith("_")]
    param_vals = [space[k] for k in param_keys]

    variants = []
    for combo in itertools.product(*param_vals):
        params = dict(zip(param_keys, combo))

        # Prune: matmul needs square tiles ≤ block_size
        if kernel == "matmul":
            if params["tile_x"] != params["tile_y"]:
                continue
            if params["tile_x"] > params["block_size"]:
                continue

        tag  = _variant_tag(kernel, params)
        path = GEN_DIR / f"{tag}.cu"
        variants.append((params, path))

    return variants


def write_variant(kernel: str, params: dict, out_path: Path) -> None:
    gen = GENERATORS.get(kernel)
    if gen is None:
        raise ValueError(f"No generator for kernel '{kernel}'")
    src = gen(params)
    out_path.write_text(src, encoding="utf-8")


if __name__ == "__main__":
    import sys
    kernel = sys.argv[1] if len(sys.argv) > 1 else "matmul"

    # Build a dummy profile to get the search space
    from parser import KernelProfile, MemoryAccessPattern, build_search_space
    dummy = KernelProfile(
        name=kernel, src_path=Path("."),
        block_dim=16, uses_shared=False,
        loop_depth=2, reduction_ops=["+="],
        memory=MemoryAccessPattern(
            has_strided_access=(kernel == "matmul"),
            has_reduction=(kernel == "reduction"),
        )
    )
    space    = build_search_space(dummy)
    variants = enumerate_variants(kernel, space)

    print(f"Kernel: {kernel}  →  {len(variants)} variants")
    for params, path in variants[:3]:
        write_variant(kernel, params, path)
        print(f"  Written: {path.name}")
    if len(variants) > 3:
        print(f"  ... and {len(variants)-3} more")
