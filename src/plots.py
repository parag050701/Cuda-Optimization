"""
plots.py — generate figures from the auto-tuner's JSON artifacts.

Reads:
  results/baseline.json
  results/{matmul,softmax,reduction,layernorm}_tuning.json

Writes (into results/figures/):
  speedup_bars.png       bar chart of baseline_ms vs best_ms per kernel
  speedup_factor.png     speedup factor per kernel with 3x target line
  roofline.png           roofline plot — kernels as points over peak lines

Only needs matplotlib — no GPU or CUDA required.

    python src/plots.py              # uses default results/ directory
    python src/plots.py <dir>        # read from a custom directory
"""

import json
import math
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")                    # headless / no display
import matplotlib.pyplot as plt

from roofline import RTX2070, KERNEL_COUNTS, compute_roofline


KERNELS = ["matmul", "softmax", "reduction", "layernorm"]
TARGET_SPEEDUP = 3.0


def _load(dir_: Path) -> dict:
    """Return {kernel: {baseline_ms, best_ms, roofline}} — missing kernels skipped."""
    data: dict[str, dict] = {}

    baseline = {}
    baseline_f = dir_ / "baseline.json"
    if baseline_f.exists():
        raw = json.loads(baseline_f.read_text())
        # baseline.json is {kernel_key: mean_ms} (old format) or a list (new)
        if isinstance(raw, dict):
            baseline = raw

    for k in KERNELS:
        f = dir_ / f"{k}_tuning.json"
        if not f.exists():
            continue
        tj = json.loads(f.read_text())
        best = tj.get("best") or {}
        if "mean_ms" not in best:
            continue
        base_ms = tj.get("baseline_ms")
        if base_ms is None:
            base_ms = next(
                (v for key, v in baseline.items() if key.startswith(k)), None
            )
        data[k] = {
            "baseline_ms": base_ms,
            "best_ms":     best["mean_ms"],
            "roofline":    tj.get("roofline"),
            "ncu_metrics": tj.get("ncu_metrics") or {},
        }
    return data


# ── Plot 1: side-by-side ms bars ──────────────────────────────────────────

def plot_ms_bars(data: dict, out: Path):
    kernels = [k for k in KERNELS if k in data]
    if not kernels:
        return
    x = range(len(kernels))
    base = [data[k]["baseline_ms"] or 0.0 for k in kernels]
    best = [data[k]["best_ms"]               for k in kernels]

    fig, ax = plt.subplots(figsize=(7, 4))
    w = 0.38
    ax.bar([i - w/2 for i in x], base, width=w, label="Baseline",
           color="#c44", edgecolor="black")
    ax.bar([i + w/2 for i in x], best, width=w, label="Best tuned",
           color="#4a7", edgecolor="black")

    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(kernels)
    ax.set_ylabel("Time per kernel (ms, log scale)")
    ax.set_title(f"Per-kernel timing on {RTX2070['name']}")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, which="both")

    for i, (b, t) in enumerate(zip(base, best)):
        if b:
            ax.text(i - w/2, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w/2, t, f"{t:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ── Plot 2: speedup factor bar chart with 3x target line ──────────────────

def plot_speedup(data: dict, out: Path):
    kernels = [k for k in KERNELS if k in data and data[k]["baseline_ms"]]
    if not kernels:
        return

    speedups = [data[k]["baseline_ms"] / data[k]["best_ms"] for k in kernels]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(kernels, speedups,
                  color=["#4a7" if s >= TARGET_SPEEDUP else "#ca4"
                         for s in speedups],
                  edgecolor="black")
    ax.axhline(TARGET_SPEEDUP, color="red", linestyle="--",
               label=f"{TARGET_SPEEDUP:.0f}× target")

    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{s:.2f}×",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Speedup (baseline / best)")
    ax.set_title(f"Auto-tuner speedup over naive baseline — {RTX2070['name']}")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ── Plot 3: roofline ───────────────────────────────────────────────────────

def plot_roofline(data: dict, out: Path):
    """
    Log-log roofline: x = arithmetic intensity (flop/byte),
    y = throughput (GFLOPS). Two lines: memory-bound ceiling
    (AI * peak_BW) and compute-bound ceiling (peak_GFLOPS).
    """
    peak_flops = RTX2070["peak_fp32_tflops"] * 1000      # GFLOPS
    peak_bw    = RTX2070["peak_bw_gbs"]

    kernels = [k for k in KERNELS if k in data]
    if not kernels:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    # Roofline itself: piecewise, crossing at AI* = peak_flops / peak_bw
    ai_knee = peak_flops / peak_bw
    ai_grid = [10 ** (i / 4) for i in range(-12, 20)]    # 1e-3 .. 1e5
    roof    = [min(peak_flops, ai * peak_bw) for ai in ai_grid]
    ax.plot(ai_grid, roof, color="black", linewidth=1.5, label="Roofline")

    ax.axvline(ai_knee, color="grey", linestyle=":", linewidth=0.8)
    ax.text(ai_knee * 1.1, peak_flops * 0.6,
            f"knee AI = {ai_knee:.1f}", fontsize=8, color="grey")

    # Plot each kernel's analytic AI vs achieved GFLOPS
    colors = {"matmul": "#c44", "softmax": "#4a7",
              "reduction": "#47c", "layernorm": "#c4a"}
    for k in kernels:
        rl = data[k]["roofline"]
        if rl is None:
            continue
        ai_a   = rl["ai_analytic"]
        gf_a   = rl["gflops_analytic"]
        ax.plot(ai_a, gf_a, marker="o", markersize=9,
                color=colors.get(k, "black"),
                label=f"{k} ({rl['pct_of_roof']:.0f}% of roof)")

        # If ncu measured AI/BW available, plot as an open marker
        ai_m = rl.get("ai_measured")
        bw_m = rl.get("bw_measured_gbs")
        if ai_m is not None and bw_m is not None:
            gf_m = gf_a   # measured flops same as analytic (ncu flop counter differs)
            ax.plot(ai_m, gf_m, marker="s", markersize=8,
                    markerfacecolor="none", markeredgecolor=colors.get(k, "black"),
                    markeredgewidth=1.5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.05, 10000)
    ax.set_ylim(1, peak_flops * 2)
    ax.set_xlabel("Arithmetic intensity (flop/byte)")
    ax.set_ylabel("Throughput (GFLOPS)")
    ax.set_title(f"Roofline — {RTX2070['name']} "
                 f"(peak {RTX2070['peak_fp32_tflops']:.1f} TFLOPS / "
                 f"{RTX2070['peak_bw_gbs']:.0f} GB/s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ── Entry point ────────────────────────────────────────────────────────────

def main(results_dir: Path):
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = _load(results_dir)
    if not data:
        print(f"No tuning JSON files found in {results_dir}")
        return

    plot_ms_bars (data, out_dir / "ms_bars.png")
    plot_speedup (data, out_dir / "speedup_factor.png")
    plot_roofline(data, out_dir / "roofline.png")

    print(f"Wrote {len(list(out_dir.glob('*.png')))} figure(s) to {out_dir}")


if __name__ == "__main__":
    rd = Path(sys.argv[1]) if len(sys.argv) > 1 else \
         Path(__file__).parent.parent / "results"
    main(rd)
