"""
autotune.py — main entrypoint for the CUDA kernel auto-tuner.

Usage:
    python autotune.py --kernel=matmul --size=1024
    python autotune.py --kernel=reduction --iters=200
    python autotune.py --baseline-only

Pipeline:
    1. Parse baseline kernel → extract profile + search space
    2. Generate all variant .cu files
    3. Compile each with nvcc (parallel jobs)
    4. Benchmark each → collect mean_ms
    5. Report best config + speedup over baseline
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT        = Path(__file__).parent
SRC_DIR     = ROOT / "src"
RESULTS_DIR = ROOT / "results"
GEN_DIR     = RESULTS_DIR / "generated"
BINS_DIR    = RESULTS_DIR / "bins"
GEN_DIR.mkdir(parents=True, exist_ok=True)
BINS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from parser    import KernelProfile, MemoryAccessPattern, build_search_space
from generator import enumerate_variants, write_variant, ARCH

WARMUP = 5
ITERS  = 100


# ── Compilation ─────────────────────────────────────────────────────────────

def compile_variant(src: Path, binary: Path) -> tuple[bool, str]:
    cmd = [
        "nvcc", "-O3", f"-arch={ARCH}",
        "--use_fast_math",
        str(src), "-o", str(binary),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return r.returncode == 0, r.stderr


def run_variant(binary: Path) -> float | None:
    """Run a compiled benchmark binary, return mean_ms or None on failure."""
    env = os.environ.copy()
    env["WARMUP"] = str(WARMUP)
    env["ITERS"]  = str(ITERS)
    try:
        r = subprocess.run([str(binary)], capture_output=True, text=True,
                           env=env, timeout=120)
        for line in r.stdout.splitlines():
            if line.startswith("TIMING"):
                parts = line.split()
                return float(parts[2])   # mean_ms
    except subprocess.TimeoutExpired:
        pass
    return None


# ── Baseline ─────────────────────────────────────────────────────────────────

def run_baseline() -> dict[str, float]:
    """Compile and run the baseline runner, return {kernel_name: mean_ms}."""
    src    = SRC_DIR / "kernels" / "benchmark_runner.cu"
    binary = BINS_DIR / "baseline_runner"

    print("[COMPILE] baseline_runner.cu ...", end=" ", flush=True)
    ok, stderr = compile_variant(src, binary)
    if not ok:
        print("FAILED\n" + stderr)
        return {}
    print("OK")

    env = os.environ.copy()
    env["WARMUP"] = str(WARMUP)
    env["ITERS"]  = str(ITERS)

    print(f"[RUN] baseline (warmup={WARMUP}, iters={ITERS}) ...")
    r = subprocess.run([str(binary)], capture_output=True, text=True,
                       env=env, timeout=300)
    print(r.stdout)

    timings = {}
    for line in r.stdout.splitlines():
        if line.startswith("TIMING"):
            parts = line.split()
            timings[parts[1]] = float(parts[2])

    out = RESULTS_DIR / "baseline.json"
    with open(out, "w") as f:
        json.dump(timings, f, indent=2)
    print(f"[SAVED] {out}\n")
    return timings


# ── Auto-tuning ──────────────────────────────────────────────────────────────

def build_kernel_profile(kernel: str) -> KernelProfile:
    """Build a representative profile for the given kernel name."""
    memory = MemoryAccessPattern(
        has_global_load    = True,
        has_global_store   = True,
        has_strided_access = kernel == "matmul",
        has_reduction      = kernel in ("reduction", "softmax"),
        has_shared_mem     = False,
    )
    return KernelProfile(
        name          = kernel,
        src_path      = SRC_DIR / "kernels" / "baseline_kernels.cu",
        block_dim     = 16,
        uses_shared   = False,
        loop_depth    = 2 if kernel == "matmul" else 1,
        reduction_ops = ["+="] if kernel in ("reduction", "softmax") else [],
        memory        = memory,
    )


def autotune_kernel(kernel: str, baseline_ms: float | None,
                    max_workers: int = 4) -> dict:
    profile  = build_kernel_profile(kernel)
    space    = build_search_space(profile)
    variants = enumerate_variants(kernel, space)

    total = len(variants)
    print(f"\n[GENERATE] {total} variants for '{kernel}' ...")
    for params, src_path in variants:
        write_variant(kernel, params, src_path)

    print(f"[COMPILE+BENCHMARK] {total} variants "
          f"(warmup={WARMUP}, iters={ITERS}) — this may take a while ...")

    results = []
    done    = 0
    t0      = time.time()

    def process(item):
        params, src_path = item
        binary = BINS_DIR / (src_path.stem + ".exe")
        ok, _  = compile_variant(src_path, binary)
        if not ok:
            return params, None
        mean_ms = run_variant(binary)
        return params, mean_ms

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process, v): v for v in variants}
        for fut in as_completed(futures):
            params, mean_ms = fut.result()
            done += 1
            elapsed = time.time() - t0
            eta     = elapsed / done * (total - done)
            status  = f"{mean_ms:.3f}ms" if mean_ms else "FAIL"
            print(f"  [{done:3d}/{total}] {status:<12} ETA {eta:.0f}s", end="\r")
            if mean_ms is not None:
                results.append({"params": params, "mean_ms": mean_ms})

    print()  # newline after \r

    if not results:
        print("[ERROR] No variants compiled successfully.")
        return {}

    results.sort(key=lambda x: x["mean_ms"])
    best = results[0]

    out_path = RESULTS_DIR / f"{kernel}_tuning.json"
    with open(out_path, "w") as f:
        json.dump({"kernel": kernel, "variants": results,
                   "best": best, "baseline_ms": baseline_ms}, f, indent=2)

    speedup = (baseline_ms / best["mean_ms"]) if baseline_ms else None

    print(f"\n{'='*60}")
    print(f"KERNEL:   {kernel}")
    print(f"BEST:     {best['params']}")
    print(f"TIME:     {best['mean_ms']:.3f}ms")
    if speedup:
        print(f"BASELINE: {baseline_ms:.3f}ms  →  SPEEDUP: {speedup:.2f}x")
    print(f"RESULTS:  {out_path}")
    print('='*60)

    return best


# ── CLI ──────────────────────────────────────────────────────────────────────

SUPPORTED_KERNELS = ["matmul", "softmax", "reduction"]

def main():
    parser = argparse.ArgumentParser(
        description="CUDA kernel auto-tuner for RTX 2070"
    )
    parser.add_argument("--kernel",   choices=SUPPORTED_KERNELS + ["all"],
                        default="matmul",
                        help="Which kernel to tune (default: matmul)")
    parser.add_argument("--baseline-only", action="store_true",
                        help="Only run baseline benchmarks, skip tuning")
    parser.add_argument("--workers",  type=int, default=2,
                        help="Parallel compile+run workers (default: 2)")
    parser.add_argument("--warmup",   type=int, default=5)
    parser.add_argument("--iters",    type=int, default=100)
    args = parser.parse_args()

    global WARMUP, ITERS
    WARMUP = args.warmup
    ITERS  = args.iters

    print("╔══════════════════════════════════════════╗")
    print("║  CUDA Kernel Auto-Tuner  |  RTX 2070     ║")
    print("╚══════════════════════════════════════════╝\n")

    baselines = run_baseline()

    if args.baseline_only:
        return

    kernels = SUPPORTED_KERNELS if args.kernel == "all" else [args.kernel]

    all_results = {}
    for k in kernels:
        # Match baseline key prefix
        baseline_ms = next(
            (v for key, v in baselines.items() if key.startswith(k)), None
        )
        best = autotune_kernel(k, baseline_ms, max_workers=args.workers)
        all_results[k] = best

    print("\n[DONE] Full summary:")
    for k, best in all_results.items():
        baseline_ms = next(
            (v for key, v in baselines.items() if key.startswith(k)), None
        )
        if best and baseline_ms:
            speedup = baseline_ms / best["mean_ms"]
            print(f"  {k:<12} {baseline_ms:.3f}ms → {best['mean_ms']:.3f}ms  "
                  f"({speedup:.2f}x speedup)")


if __name__ == "__main__":
    main()
