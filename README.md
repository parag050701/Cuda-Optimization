# CUDA Kernel Auto-Tuner

A source-to-source CUDA compiler and auto-tuner that takes naive kernels and searches a space of optimized variants — tiling, register blocking, loop unrolling, warp shuffles, memory-access reordering — then picks the fastest variant that produces numerically correct output.

**Target hardware:** NVIDIA RTX 2070 (Turing, sm_75) — 2304 cores, 8 GB GDDR6, 448 GB/s peak bandwidth, 7.46 TFLOPS peak fp32.

Four kernels are targeted, spanning compute- and memory-bound regimes:

| Kernel | Problem size | Characteristic |
|---|---|---|
| Matrix multiplication | 1024×1024 fp32 | Compute-bound (AI ≈ 170 flop/byte) |
| Softmax | 1024×4096 | Memory-bound with row-wise reduction |
| Parallel reduction | 2²⁰ elements | Memory-bound, classic GPU micro-benchmark |
| Layer normalization | 512×2048 | Memory-bound with fused mean/variance |

## Pipeline

```
    baseline_kernels.cu
           │
           ▼  parser.py (libclang AST walker)
    KernelProfile ─── build_search_space ───► parameter grid
                                                   │
                                                   ▼
                                          generator.py (templates)
                                                   │
                                                   ▼
                                            variant .cu files
                                                   │
                                                   ▼
                                           nvcc + run (autotune.py)
                                                   │
                                                   ▼
                                      TIMING + CHECK (correctness)
                                                   │
                                                   ▼
                                    ncu metrics (Nsight Compute)
                                                   │
                                                   ▼
                                    roofline.py → results/*.json
                                                   │
                                                   ▼
                                        plots.py → figures
```

Each generated binary embeds the naive reference kernel, runs it on the same inputs, and emits a `CHECK` line (`max_rel_err` and pass/fail) alongside the `TIMING` line. Variants that exceed 1% relative error are dropped automatically.

## Quick start

```bash
# 1. Install Python deps (libclang ships its own shared library, no system install needed)
pip install -r requirements.txt

# 2. CUDA toolkit must be installed and nvcc/ncu on PATH
nvcc --version
ncu --version         # optional; autotune.py runs without it if --skip-ncu

# 3. Run the full pipeline (baseline + all four kernels)
python autotune.py --kernel=all --warmup=5 --iters=100 --workers=2

# 4. Render the writeup figures
python src/plots.py
```

Artifacts land in `results/`:
- `baseline.json` — naive kernel timings
- `<kernel>_tuning.json` — full variant list, `incorrect` (correctness-failed) list, `best`, `ncu_metrics`, `roofline`
- `figures/ms_bars.png`, `figures/speedup_factor.png`, `figures/roofline.png`

## Individual components

```bash
# Parse a .cu file and print detected kernels + search spaces
python src/parser.py [src/kernels/baseline_kernels.cu]

# Generate a few variants of one kernel type
python src/generator.py matmul

# Print a roofline table from saved tuning JSONs
python src/roofline.py

# Re-render plots (no GPU required)
python src/plots.py
```

## Project structure

```
autotune.py                     CLI driver — baseline, tune, ncu, roofline
requirements.txt

src/
  parser.py                     libclang AST walker (+ regex fallback)
  generator.py                  kernel templates + correctness harness
  benchmark.py                  nvcc wrapper, ncu metric collection
  roofline.py                   RTX 2070 constants + roofline math
  plots.py                      matplotlib figures from tuning JSONs

  kernels/
    baseline_kernels.cu         naive reference implementations (source-
                                of-truth for the parser)
    naive_kernels.cuh           header form of the naive kernels; every
                                generated variant includes this for its
                                correctness comparison
    benchmark_runner.cu         standalone baseline-timings binary

results/                        produced at runtime (git-ignored)
  baseline.json
  <kernel>_tuning.json
  generated/                    one .cu per variant
  bins/                         compiled binaries
  figures/                      rendered PNGs
```

## What gets tuned

The search space is built per kernel by `parser.build_search_space` from the AST analysis. Pruning rules eliminate combinations that the template wouldn't use:

| Parameter | Values | Applies to |
|---|---|---|
| `block_size` | 64, 128, 192, 256 | softmax, reduction, layernorm |
| `tile_x`, `tile_y` | 16, 32 (forced equal) | matmul |
| `reg_tile` | 1, 2, 4 | matmul (per-thread output tile) |
| `unroll` | 1, 2, 4, 8 | all |
| `transpose_b` | true/false | matmul with strided access |
| `warp_shuffle` | true/false | reduction |

After pruning: 32 matmul + 16 softmax + 32 reduction + 16 layernorm = **96 variants total**.

## Correctness

Every generated binary runs the naive reference kernel on the same inputs as the optimized kernel, then compares outputs on the host:

- **matmul / softmax / layernorm** — element-wise, `max(|a - b| / max(|a|, |b|))` compared against a 1e-2 tolerance.
- **reduction** — scalar-sum compare (grid sizes differ between naive and double-loaded optimized, so element-wise doesn't apply).

Failed variants are excluded from the "best" selection and written to an `incorrect` list in the tuning JSON for inspection.

## Roofline analysis

`roofline.py` computes, per kernel:
- Arithmetic intensity (analytic, from problem size)
- Achieved GFLOPS and effective bandwidth at the best time
- Upper bound = `min(peak_fp32, AI × peak_bandwidth)`
- % of roofline achieved

When `ncu` is available, DRAM bytes from `dram__bytes.sum` give a second set of measured numbers (typically higher bandwidth and lower AI than the analytic lower bound, reflecting actual cache-miss traffic).

## Architecture decisions

- **libclang over a handwritten parser.** AST traversal accurately detects CUDA features (`__global__`, `__shared__`, `__shfl_*`, loop nesting) where regex heuristics over-count sequential loops as nested and miss subscripted reductions. A regex path is kept as a fallback so the tuner degrades gracefully when libclang isn't available.
- **In-process correctness.** Comparing against a naive reference kernel inside the same binary sidesteps any cross-process seed/state drift and keeps the search loop self-contained.
- **Grid search over random/Bayesian.** With 96 variants the search is small enough that exhaustive grid dominates on both wall-clock and simplicity.
- **No tensor-core / WMMA path.** Turing's sm_75 only supports fp16 WMMA, not fp32. A meaningful WMMA implementation would be a separate project.

## Dependencies

- Python 3.10+
- `libclang>=16.0.0` (PyPI; bundles its own shared library)
- `matplotlib>=3.7`
- CUDA toolkit (`nvcc`) — tested with CUDA 12.x on sm_75
- `ncu` (Nsight Compute) — optional; auto-skipped if not on PATH

## Status

All four project phases are code-complete:

| Phase | Scope |
|---|---|
| A | libclang compiler frontend |
| B | Optimization backend + in-process correctness check |
| C | Nsight Compute integration + roofline analysis |
| D | Figure generation (speedup bars, roofline plot) |

Out of scope: tensor-core (WMMA), fused FlashAttention-style softmax, multi-GPU, float4 vectorized loads (can be added if matmul doesn't clear 3× on the final tight run).
