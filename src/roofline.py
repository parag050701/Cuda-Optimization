"""
roofline.py — roofline analysis for the auto-tuner.

Hardware target: NVIDIA RTX 2070 (Turing, sm_75).
  - 2304 CUDA cores
  - 1.62 GHz boost clock
  - Theoretical peak fp32 FMA: 2 * 2304 * 1.62e9 ≈ 7.46 TFLOPS
  - Memory: 8 GB GDDR6, 256-bit bus, 14 Gbps → 448 GB/s peak

For each kernel we record:
  - Analytic FLOPs and bytes (problem-size lower bound, ignoring cache reuse).
  - Measured GFLOPS and effective bandwidth at the best tuned time.
  - Arithmetic intensity (flops / bytes).
  - Roofline upper bound = min(peak_fp32, AI * peak_bw).
  - Percent of roofline achieved.

If ncu metrics are also supplied (dram__bytes.sum), we derive a *measured*
bytes figure that accounts for actual DRAM traffic (typically much higher
than the analytic minimum because of multi-read / cache-miss behaviour).
"""

from dataclasses import dataclass, asdict
from typing import Optional


# ── RTX 2070 peaks ─────────────────────────────────────────────────────────

RTX2070 = {
    "name":            "NVIDIA GeForce RTX 2070",
    "arch":            "sm_75",
    "cores":           2304,
    "boost_clock_ghz": 1.62,
    "peak_fp32_tflops": 7.46,            # 2 * 2304 * 1.62 GHz (FMA counted as 2)
    "peak_bw_gbs":     448.0,            # 256-bit @ 14 Gbps GDDR6
}


# ── Per-kernel analytic FLOP and byte counts ───────────────────────────────
# These use the same problem sizes as benchmark_runner.cu and the generator.

def _matmul_counts(N: int = 1024) -> dict:
    # Naive: 2N³ fp32 FMAs (counted as 2 flops each).
    # Byte lower bound: each element of A, B read once; C written once.
    return {
        "flops":  2 * (N ** 3),
        "bytes":  3 * (N ** 2) * 4,
        "shape":  f"{N}x{N}",
    }


def _softmax_counts(rows: int = 1024, cols: int = 4096) -> dict:
    # Per element: 1 subtract + 1 exp (~20 flops amortized) + 1 divide
    # Conservative: 5 flops/elem.  Bytes: read once + write once → 2 * rows*cols.
    n = rows * cols
    return {
        "flops":  5 * n,
        "bytes":  2 * n * 4,
        "shape":  f"{rows}x{cols}",
    }


def _reduction_counts(N: int = 1 << 20) -> dict:
    # One add per element; output is a single scalar (per block, but analytic
    # bytes uses the dominant input traffic).
    return {
        "flops":  N,
        "bytes":  N * 4,
        "shape":  f"N={N}",
    }


def _layernorm_counts(rows: int = 512, cols: int = 2048) -> dict:
    # Per element: mean-sum, sqr-sum, normalize, scale+shift → ~8 flops/elem.
    # Bytes: read input + gamma + beta, write output → 3 * rows*cols + 2*cols.
    n = rows * cols
    return {
        "flops":  8 * n,
        "bytes":  3 * n * 4 + 2 * cols * 4,
        "shape":  f"{rows}x{cols}",
    }


KERNEL_COUNTS = {
    "matmul":    _matmul_counts(),
    "softmax":   _softmax_counts(),
    "reduction": _reduction_counts(),
    "layernorm": _layernorm_counts(),
}


# ── Roofline calculation ───────────────────────────────────────────────────

@dataclass
class RooflineResult:
    kernel:            str
    shape:             str
    mean_ms:           float

    # Analytic (problem-size lower bounds)
    flops_analytic:    float
    bytes_analytic:    float
    ai_analytic:       float        # flop/byte
    gflops_analytic:   float
    bw_analytic_gbs:   float

    # Roofline bound at the analytic arithmetic intensity
    bound_gflops:      float        # min(peak_fp32, ai * peak_bw)
    pct_of_roof:       float        # measured_gflops / bound_gflops * 100

    # Optional: measured DRAM bytes from ncu (dram__bytes.sum)
    bytes_measured:    Optional[float] = None
    bw_measured_gbs:   Optional[float] = None
    ai_measured:       Optional[float] = None


def compute_roofline(kernel: str, mean_ms: float,
                     ncu_metrics: Optional[dict] = None) -> RooflineResult:
    """Compute roofline stats for a kernel at a given measured runtime."""
    counts = KERNEL_COUNTS[kernel]
    flops  = counts["flops"]
    bytes_ = counts["bytes"]

    time_s = mean_ms * 1e-3
    gflops = flops  / time_s / 1e9
    bw_gbs = bytes_ / time_s / 1e9
    ai     = flops  / bytes_

    peak_flops_gflops = RTX2070["peak_fp32_tflops"] * 1000
    peak_bw           = RTX2070["peak_bw_gbs"]
    bound             = min(peak_flops_gflops, ai * peak_bw)
    pct               = 100.0 * gflops / bound if bound > 0 else 0.0

    bytes_meas = bw_meas = ai_meas = None
    if ncu_metrics:
        dram = ncu_metrics.get("dram__bytes.sum")
        if dram is not None and dram > 0:
            bytes_meas = float(dram)
            bw_meas    = bytes_meas / time_s / 1e9
            ai_meas    = flops / bytes_meas

    return RooflineResult(
        kernel           = kernel,
        shape            = counts["shape"],
        mean_ms          = mean_ms,
        flops_analytic   = flops,
        bytes_analytic   = bytes_,
        ai_analytic      = ai,
        gflops_analytic  = gflops,
        bw_analytic_gbs  = bw_gbs,
        bound_gflops     = bound,
        pct_of_roof      = pct,
        bytes_measured   = bytes_meas,
        bw_measured_gbs  = bw_meas,
        ai_measured      = ai_meas,
    )


def format_table(results: list[RooflineResult]) -> str:
    """Return a human-readable roofline summary table."""
    lines = []
    lines.append(f"{'='*92}")
    lines.append(f"Roofline summary — {RTX2070['name']} "
                 f"(peak {RTX2070['peak_fp32_tflops']:.2f} TFLOPS / "
                 f"{RTX2070['peak_bw_gbs']:.0f} GB/s)")
    lines.append(f"{'-'*92}")
    lines.append(f"{'Kernel':<10} {'Shape':<14} {'ms':>8} "
                 f"{'GFLOPS':>9} {'BW':>8} {'AI':>7} {'% roof':>8}")
    lines.append(f"{'-'*92}")
    for r in results:
        lines.append(
            f"{r.kernel:<10} {r.shape:<14} {r.mean_ms:>8.3f} "
            f"{r.gflops_analytic:>9.1f} {r.bw_analytic_gbs:>8.1f} "
            f"{r.ai_analytic:>7.2f} {r.pct_of_roof:>7.1f}%"
        )
        if r.bw_measured_gbs is not None:
            lines.append(
                f"{'  (ncu)':<10} {'':<14} {'':<8} "
                f"{'':<9} {r.bw_measured_gbs:>8.1f} "
                f"{r.ai_measured:>7.2f}"
            )
    lines.append(f"{'='*92}")
    return "\n".join(lines)


# ── CLI: read a tuning JSON and print roofline ─────────────────────────────

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        results_dir = Path(__file__).parent.parent / "results"
    else:
        results_dir = Path(sys.argv[1])

    rr = []
    for kernel in ("matmul", "softmax", "reduction", "layernorm"):
        f = results_dir / f"{kernel}_tuning.json"
        if not f.exists():
            continue
        data = json.loads(f.read_text())
        best = data.get("best")
        if not best:
            continue
        ncu = data.get("ncu_metrics") or {}
        rr.append(compute_roofline(kernel, best["mean_ms"], ncu))

    if rr:
        print(format_table(rr))
    else:
        print(f"No tuning JSON files found under {results_dir}")
