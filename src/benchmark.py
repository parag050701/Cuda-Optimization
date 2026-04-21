"""
benchmark.py — compile, run, and profile CUDA kernels.

Wraps nvcc compilation and CUDA event timing. Outputs JSON metrics
to results/. Also queries nvidia-smi for memory throughput when
Nsight Compute (ncu) is available.
"""

import subprocess
import json
import os
import sys
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Compilation ────────────────────────────────────────────────────────────

def compile_kernel(src_path: Path, out_path: Path,
                   arch: str = "sm_75",
                   extra_flags: list[str] | None = None) -> tuple[bool, str]:
    """Compile a .cu file with nvcc. Returns (success, stderr)."""
    flags = [
        "nvcc",
        "-O3",
        f"-arch={arch}",
        "--use_fast_math",
        "-Xptxas", "-v",           # verbose PTX stats (registers, smem)
        str(src_path),
        "-o", str(out_path),
    ]
    if extra_flags:
        flags.extend(extra_flags)

    result = subprocess.run(flags, capture_output=True, text=True)
    return result.returncode == 0, result.stderr


# ── Benchmarking ───────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    kernel: str
    variant: str
    params: dict
    mean_ms: float
    min_ms: float
    max_ms: float
    iters: int
    compile_ok: bool
    ptx_info: str          # register/smem counts from nvcc -Xptxas -v
    ncu_metrics: dict      # Nsight Compute metrics if available


def run_binary(binary: Path, env_overrides: dict | None = None,
               timeout: int = 60) -> tuple[bool, str, str]:
    """Run a compiled binary, return (success, stdout, stderr)."""
    env = os.environ.copy()
    if env_overrides:
        env.update({k: str(v) for k, v in env_overrides.items()})

    result = subprocess.run(
        [str(binary)],
        capture_output=True, text=True,
        env=env, timeout=timeout
    )
    return result.returncode == 0, result.stdout, result.stderr


def collect_ncu_metrics(binary: Path, metrics: list[str]) -> dict:
    """
    Run Nsight Compute (ncu) to collect hardware counters.
    Returns empty dict if ncu is not installed or fails.
    Key metrics for memory-bound analysis:
      - l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum  (global loads)
      - dram__bytes.sum                                (DRAM bandwidth)
      - smsp__sass_thread_inst_executed_op_fadd_pred_on.sum (FP ops)
    """
    if not metrics:
        metrics = [
            "dram__bytes.sum",
            "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum",
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "sm__occupancy_max_warps_active.avg.pct_of_peak_sustained_active",
        ]

    metric_str = ",".join(metrics)
    cmd = [
        "ncu",
        "--metrics", metric_str,
        "--csv",
        "--target-processes", "all",
        str(binary),
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return {}
        return _parse_ncu_csv(result.stdout)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return {}


def _parse_ncu_csv(csv_text: str) -> dict:
    """Parse ncu --csv output into {metric_name: value} dict."""
    metrics = {}
    lines = csv_text.strip().splitlines()
    if len(lines) < 2:
        return metrics

    headers = [h.strip('"') for h in lines[0].split(",")]
    for line in lines[1:]:
        parts = [p.strip('"') for p in line.split(",")]
        if len(parts) < len(headers):
            continue
        row = dict(zip(headers, parts))
        name = row.get("Metric Name", "")
        val  = row.get("Metric Value", "")
        if name:
            try:
                metrics[name] = float(val.replace(",", ""))
            except ValueError:
                metrics[name] = val
    return metrics


# ── Harness driver (used by benchmark_runner.cu output) ────────────────────

def parse_timing_output(stdout: str) -> dict[str, float]:
    """
    Parse timing lines from our C benchmark binaries.
    Expected format (one per line):
        TIMING kernel_name mean_ms min_ms max_ms iters
    """
    timings = {}
    for line in stdout.splitlines():
        if line.startswith("TIMING"):
            parts = line.split()
            if len(parts) >= 5:
                name    = parts[1]
                mean_ms = float(parts[2])
                min_ms  = float(parts[3])
                max_ms  = float(parts[4])
                iters   = int(parts[5]) if len(parts) > 5 else 0
                timings[name] = {
                    "mean_ms": mean_ms,
                    "min_ms":  min_ms,
                    "max_ms":  max_ms,
                    "iters":   iters,
                }
    return timings


# ── Main baseline benchmark ────────────────────────────────────────────────

def run_baseline_benchmark(warmup: int = 5, iters: int = 100) -> list[dict]:
    """Compile and run the baseline benchmark binary, save results."""
    src   = ROOT / "src" / "kernels" / "benchmark_runner.cu"
    binary = ROOT / "results" / "baseline_runner"

    print(f"[COMPILE] {src.name} ...", end=" ", flush=True)
    ok, stderr = compile_kernel(src, binary)
    if not ok:
        print("FAILED")
        print(stderr)
        return []
    print("OK")

    # Extract PTX info (register count, smem usage per kernel)
    ptx_info = "\n".join(
        l for l in stderr.splitlines() if "ptxas info" in l.lower()
    )

    print(f"[RUN] warmup={warmup} iters={iters} ...")
    env = {"WARMUP": warmup, "ITERS": iters}
    ok, stdout, run_stderr = run_binary(binary, env)
    if not ok:
        print("BINARY FAILED:", run_stderr)
        return []

    timings = parse_timing_output(stdout)
    print(stdout)

    # Optionally collect Nsight Compute metrics
    print("[NCU] collecting hardware counters (skip with Ctrl+C) ...")
    ncu = {}
    try:
        ncu = collect_ncu_metrics(binary, [])
    except KeyboardInterrupt:
        print("  skipped.")

    results = []
    for kernel, t in timings.items():
        r = {
            "kernel":      kernel,
            "variant":     "baseline",
            "params":      {"warmup": warmup, "iters": iters},
            "mean_ms":     t["mean_ms"],
            "min_ms":      t["min_ms"],
            "max_ms":      t["max_ms"],
            "iters":       t["iters"],
            "compile_ok":  True,
            "ptx_info":    ptx_info,
            "ncu_metrics": ncu,
        }
        results.append(r)

    out_path = RESULTS_DIR / "baseline.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[SAVED] {out_path}")
    return results


def print_summary(results: list[dict]) -> None:
    print("\n" + "="*60)
    print(f"{'Kernel':<25} {'Mean (ms)':>10} {'Min (ms)':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['kernel']:<25} {r['mean_ms']:>10.3f} {r['min_ms']:>10.3f}")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline CUDA kernel benchmarker")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters",  type=int, default=100)
    args = parser.parse_args()

    results = run_baseline_benchmark(args.warmup, args.iters)
    if results:
        print_summary(results)
