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
from dataclasses import asdict
from pathlib import Path

ROOT        = Path(__file__).parent
SRC_DIR     = ROOT / "src"
RESULTS_DIR = ROOT / "results"
GEN_DIR     = RESULTS_DIR / "generated"
BINS_DIR    = RESULTS_DIR / "bins"
GEN_DIR.mkdir(parents=True, exist_ok=True)
BINS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(SRC_DIR))

from parser     import (KernelProfile, MemoryAccessPattern,
                        build_search_space, find_kernel)
from generator  import enumerate_variants, write_variant, ARCH
from benchmark  import collect_ncu_metrics
from roofline   import compute_roofline, format_table, RooflineResult

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


def run_variant(binary: Path) -> dict | None:
    """
    Run a compiled benchmark binary and parse its TIMING/CHECK output.

    Returns dict with keys: mean_ms, max_rel_err, pass.
    Returns None if the run times out or neither line is produced.
    """
    env = os.environ.copy()
    env["WARMUP"] = str(WARMUP)
    env["ITERS"]  = str(ITERS)

    mean_ms: float | None   = None
    max_rel_err: float | None = None
    check_pass: bool | None = None

    try:
        r = subprocess.run([str(binary)], capture_output=True, text=True,
                           env=env, timeout=120)
    except subprocess.TimeoutExpired:
        return None

    for line in r.stdout.splitlines():
        parts = line.split()
        if line.startswith("TIMING") and len(parts) >= 3:
            try:
                mean_ms = float(parts[2])
            except ValueError:
                pass
        elif line.startswith("CHECK") and len(parts) >= 4:
            try:
                max_rel_err = float(parts[2])
                check_pass  = parts[3] == "1"
            except ValueError:
                pass

    if mean_ms is None:
        return None
    return {"mean_ms": mean_ms,
            "max_rel_err": max_rel_err,
            "pass": check_pass}


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

_BASELINE_SRC = SRC_DIR / "kernels" / "baseline_kernels.cu"


def _hardcoded_profile(kernel: str) -> KernelProfile:
    """Kernel-name-based fallback used when source parsing fails."""
    memory = MemoryAccessPattern(
        has_global_load    = True,
        has_global_store   = True,
        has_strided_access = kernel == "matmul",
        has_reduction      = kernel in ("reduction", "softmax"),
        has_shared_mem     = False,
    )
    return KernelProfile(
        name          = kernel,
        src_path      = _BASELINE_SRC,
        block_dim     = 16,
        uses_shared   = False,
        loop_depth    = 2 if kernel == "matmul" else 1,
        reduction_ops = ["+="] if kernel in ("reduction", "softmax") else [],
        memory        = memory,
        backend       = "hardcoded",
    )


def build_kernel_profile(kernel: str) -> KernelProfile:
    """
    Derive a KernelProfile by parsing the baseline source. Falls back to a
    kernel-name-based heuristic when the parser returns nothing (e.g. the
    libclang binding is missing and the regex analyzer also fails).
    """
    profile = find_kernel(_BASELINE_SRC, kernel, verbose=True)
    if profile is not None:
        print(f"[PARSE] {kernel} → {profile.name}  "
              f"[backend={profile.backend}]  "
              f"loop_depth={profile.loop_depth}  "
              f"shared={profile.uses_shared}  "
              f"strided={profile.memory.has_strided_access}  "
              f"reduction={profile.memory.has_reduction}")
        return profile

    print(f"[PARSE] {kernel}: no match in {_BASELINE_SRC.name}, "
          f"using hardcoded fallback")
    return _hardcoded_profile(kernel)


def autotune_kernel(kernel: str, baseline_ms: float | None,
                    max_workers: int = 4,
                    skip_ncu: bool = False) -> dict:
    profile  = build_kernel_profile(kernel)
    space    = build_search_space(profile)
    variants = enumerate_variants(kernel, space)

    total = len(variants)
    print(f"\n[GENERATE] {total} variants for '{kernel}' ...")
    for params, src_path in variants:
        write_variant(kernel, params, src_path)

    print(f"[COMPILE+BENCHMARK] {total} variants "
          f"(warmup={WARMUP}, iters={ITERS}) — this may take a while ...")

    results: list[dict]        = []
    incorrect: list[dict]      = []
    n_compile_fail             = 0
    done                       = 0
    t0                         = time.time()

    def process(item):
        params, src_path = item
        binary = BINS_DIR / (src_path.stem + ".exe")
        ok, _  = compile_variant(src_path, binary)
        if not ok:
            return params, str(binary), None, "compile_fail"
        info = run_variant(binary)
        if info is None:
            return params, str(binary), None, "run_fail"
        return params, str(binary), info, None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process, v): v for v in variants}
        for fut in as_completed(futures):
            params, binary_str, info, err = fut.result()
            done += 1
            elapsed = time.time() - t0
            eta     = elapsed / done * (total - done)

            if info is None:
                n_compile_fail += (err == "compile_fail")
                status = "FAIL"
            elif info.get("pass") is False:
                incorrect.append({"params": params, "binary": binary_str, **info})
                status = f"WRONG({info['max_rel_err']:.1e})"
            else:
                results.append({"params": params, "binary": binary_str, **info})
                status = f"{info['mean_ms']:.3f}ms"

            print(f"  [{done:3d}/{total}] {status:<16} ETA {eta:.0f}s", end="\r")

    print()  # newline after \r
    if incorrect:
        print(f"[CORRECTNESS] {len(incorrect)} variant(s) exceeded "
              f"tolerance and were dropped.")
    if n_compile_fail:
        print(f"[COMPILE] {n_compile_fail} variant(s) failed to compile.")

    if not results:
        print("[ERROR] No variants both compiled and passed correctness.")
        return {}

    results.sort(key=lambda x: x["mean_ms"])
    best = results[0]

    # ── Nsight Compute profiling of the best variant ────────────────────
    ncu_metrics: dict = {}
    if not skip_ncu and "binary" in best:
        best_bin = Path(best["binary"])
        if best_bin.exists():
            print(f"[NCU] profiling {best_bin.name} "
                  f"(kernel filter: .*_opt, launches=3) ...")
            # WARMUP=0, ITERS=3 keeps the profiling run short — the binary
            # still does 1 naive-ref launch + 3 optimized launches. The
            # kernel regex filters ncu to only record the optimized kernel.
            ncu_metrics = collect_ncu_metrics(
                best_bin,
                kernel_regex=".*_opt",
                launch_count=3,
                env_overrides={"WARMUP": 0, "ITERS": 3},
            )
            if ncu_metrics:
                print(f"[NCU] collected {len(ncu_metrics)} metric(s)")
            else:
                print(f"[NCU] no metrics collected (ncu may be unavailable)")

    # ── Roofline analysis ───────────────────────────────────────────────
    roofline = compute_roofline(kernel, best["mean_ms"], ncu_metrics)

    out_path = RESULTS_DIR / f"{kernel}_tuning.json"
    with open(out_path, "w") as f:
        json.dump({"kernel": kernel,
                   "variants": results,
                   "incorrect": incorrect,
                   "best": best,
                   "baseline_ms": baseline_ms,
                   "ncu_metrics": ncu_metrics,
                   "roofline": asdict(roofline)}, f, indent=2)

    speedup = (baseline_ms / best["mean_ms"]) if baseline_ms else None

    print(f"\n{'='*60}")
    print(f"KERNEL:   {kernel}")
    print(f"BEST:     {best['params']}")
    print(f"TIME:     {best['mean_ms']:.3f}ms")
    if speedup:
        print(f"BASELINE: {baseline_ms:.3f}ms  →  SPEEDUP: {speedup:.2f}x")
    print(f"GFLOPS:   {roofline.gflops_analytic:.1f}  "
          f"BW: {roofline.bw_analytic_gbs:.1f} GB/s  "
          f"({roofline.pct_of_roof:.1f}% of roofline)")
    if roofline.bw_measured_gbs is not None:
        print(f"(ncu)     measured BW: {roofline.bw_measured_gbs:.1f} GB/s  "
              f"AI: {roofline.ai_measured:.2f} flop/byte")
    print(f"RESULTS:  {out_path}")
    print('='*60)

    return {"best": best, "roofline": roofline}


# ── CLI ──────────────────────────────────────────────────────────────────────

SUPPORTED_KERNELS = ["matmul", "softmax", "reduction", "layernorm"]

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
    parser.add_argument("--skip-ncu", action="store_true",
                        help="Skip Nsight Compute profiling of best variants")
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

    all_results: dict[str, dict] = {}
    for k in kernels:
        baseline_ms = next(
            (v for key, v in baselines.items() if key.startswith(k)), None
        )
        result = autotune_kernel(k, baseline_ms,
                                 max_workers=args.workers,
                                 skip_ncu=args.skip_ncu)
        if result:
            all_results[k] = result

    if not all_results:
        return

    # Speedup summary
    print("\n[DONE] Speedup summary:")
    for k, r in all_results.items():
        baseline_ms = next(
            (v for key, v in baselines.items() if key.startswith(k)), None
        )
        best = r["best"]
        if baseline_ms:
            speedup = baseline_ms / best["mean_ms"]
            print(f"  {k:<12} {baseline_ms:.3f}ms → {best['mean_ms']:.3f}ms  "
                  f"({speedup:.2f}x speedup)")

    # Roofline table
    print()
    print(format_table([r["roofline"] for r in all_results.values()]))


if __name__ == "__main__":
    main()
