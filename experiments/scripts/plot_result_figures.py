"""Render the RQ1 and RQ2 result figures that replace the CI-bearing tables.

Three thesis tables carried 95% confidence intervals and are now figures:
  - RQ1 candidate speedup, one figure spanning both platforms (MPS top, T4 bottom)
  - RQ2 hybrid applicability ladder, one figure spanning both platforms

Every panel starts its value axis at 0 and draws a dashed parity line at 1.0x,
so a bar below the line is a regression. Bars carry asymmetric error bars from
the recorded 95% CI. Source data:
  - RQ1: projection.json -> rq1.candidates (speedup_geomean, ci_low, ci_high)
  - RQ2: experiment-2.json -> record.workloads.<wl>.ladder
         (speedup_mean, speedup_ci_low, speedup_ci_high per slowdown rung)

This reads recorded results; it runs nothing. Output is vector PDF for LaTeX.

Run from the experiments/ root:
    python scripts/plot_result_figures.py --write
    python scripts/plot_result_figures.py \
        --mps-dir outputs/platform_experiment_suite/20260601T132914Z \
        --t4-dir canonical-results/t4-results \
        --out-dir ../docs/thesis/figures --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_DEFAULT_MPS = Path("canonical-results/mps-results")
_DEFAULT_T4 = Path("canonical-results/t4-results")
_DEFAULT_OUT = Path("../docs/thesis/figures")

# Stable display order and short labels.
_WORKLOADS = [
    ("cifar10_resnet20_classification", "cifar10_resnet20"),
    ("cifar10_row_gru_classification", "cifar10_row_gru"),
    ("mnist_mlp_classification", "mnist_mlp"),
    ("california_mlp_regression", "california_mlp"),
]
_CANDIDATES = [
    ("vmapped_k32", "vmap\nk32"),
    ("vmapped_k64", "vmap\nk64"),
    ("compiled", "comp"),
    ("compiled_vmapped_k32", "c+v\nk32"),
    ("compiled_vmapped_k64", "c+v\nk64"),
]
_SLOWDOWNS = [1, 2, 4, 8, 16]

_WIN = "#2c7fb8"
_INCONCLUSIVE = "#9e9e9e"
_REGRESS = "#d7301f"
_PARITY = "#444444"


def _load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _verdict_color(ci_low: float, ci_high: float) -> str:
    """Color by where the 95% CI sits relative to parity (1.0x)."""
    if ci_low > 1.0:
        return _WIN
    if ci_high < 1.0:
        return _REGRESS
    return _INCONCLUSIVE


def _asym_err(mean: float, lo: float, hi: float) -> tuple[float, float]:
    return max(0.0, mean - lo), max(0.0, hi - mean)


def _rq1_by_workload(projection: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for cand in projection["rq1"]["candidates"]:
        out.setdefault(cand["workload"], {})[cand["candidate"]] = cand
    return out


def _rq2_ladders(exp2: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    workloads = exp2.get("record", exp2).get("workloads", {})
    return {wl: rec.get("ladder", []) for wl, rec in workloads.items()}


def _annotate(ax, x: float, top: float, mean: float) -> None:
    ax.annotate(
        f"{mean:.2g}x",
        xy=(x, top),
        xytext=(0, 2),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=6.5,
    )


def _plot_rq1(
    projections_by_platform: dict[str, dict[str, Any]],
    out_path: Path,
) -> None:
    platforms = list(projections_by_platform)
    fig, axes = plt.subplots(
        len(platforms), len(_WORKLOADS),
        figsize=(13.0, 3.1 * len(platforms)), squeeze=False,
    )
    for row, platform in enumerate(platforms):
        by_wl = _rq1_by_workload(projections_by_platform[platform])
        for col, (wl_key, wl_label) in enumerate(_WORKLOADS):
            ax = axes[row][col]
            cands = by_wl.get(wl_key, {})
            means, lows, highs, colors = [], [], [], []
            for cand_key, _ in _CANDIDATES:
                rec = cands.get(cand_key, {})
                m = rec.get("speedup_geomean", 0.0)
                lo = rec.get("ci_low", m)
                hi = rec.get("ci_high", m)
                means.append(m)
                lows.append(lo)
                highs.append(hi)
                colors.append(_verdict_color(lo, hi))
            x = range(len(_CANDIDATES))
            yerr = [[_asym_err(m, lo, hi)[0] for m, lo, hi in zip(means, lows, highs)],
                    [_asym_err(m, lo, hi)[1] for m, lo, hi in zip(means, lows, highs)]]
            ax.bar(x, means, color=colors, width=0.7)
            ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black",
                        elinewidth=0.9, capsize=2.5)
            for xi, (m, hi) in enumerate(zip(means, highs)):
                _annotate(ax, xi, hi, m)
            ax.axhline(1.0, color=_PARITY, linestyle="--", linewidth=0.9)
            ax.set_ylim(bottom=0)
            ax.set_xticks(list(x))
            ax.set_xticklabels([lbl for _, lbl in _CANDIDATES], fontsize=7)
            ax.margins(y=0.18)
            if row == 0:
                ax.set_title(wl_label, fontsize=8, fontfamily="monospace")
            if col == 0:
                ax.set_ylabel(f"{platform}\nspeedup vs vanilla", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_rq2(
    ladders_by_platform: dict[str, dict[str, list[dict[str, Any]]]],
    out_path: Path,
) -> None:
    platforms = list(ladders_by_platform)
    fig, axes = plt.subplots(
        len(platforms), len(_WORKLOADS),
        figsize=(13.0, 3.1 * len(platforms)), squeeze=False,
    )
    for row, platform in enumerate(platforms):
        ladders = ladders_by_platform[platform]
        for col, (wl_key, wl_label) in enumerate(_WORKLOADS):
            ax = axes[row][col]
            rungs = {int(r["slowdown_factor"]): r for r in ladders.get(wl_key, [])}
            means, lows, highs, colors = [], [], [], []
            for s in _SLOWDOWNS:
                r = rungs.get(s, {})
                m = r.get("speedup_mean", 0.0)
                lo = r.get("speedup_ci_low", m)
                hi = r.get("speedup_ci_high", m)
                means.append(m)
                lows.append(lo)
                highs.append(hi)
                colors.append(_verdict_color(lo, hi))
            x = range(len(_SLOWDOWNS))
            yerr = [[_asym_err(m, lo, hi)[0] for m, lo, hi in zip(means, lows, highs)],
                    [_asym_err(m, lo, hi)[1] for m, lo, hi in zip(means, lows, highs)]]
            ax.bar(x, means, color=colors, width=0.7)
            ax.errorbar(x, means, yerr=yerr, fmt="none", ecolor="black",
                        elinewidth=0.9, capsize=2.5)
            ax.axhline(1.0, color=_PARITY, linestyle="--", linewidth=0.9)
            ax.set_ylim(bottom=0)
            ax.set_xticks(list(x))
            ax.set_xticklabels([str(s) for s in _SLOWDOWNS], fontsize=7)
            ax.margins(y=0.18)
            if row == 0:
                ax.set_title(wl_label, fontsize=8, fontfamily="monospace")
            if row == len(platforms) - 1:
                ax.set_xlabel("GPU slowdown factor", fontsize=8)
            if col == 0:
                ax.set_ylabel(f"{platform}\nhybrid speedup", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mps-dir", type=Path, default=_DEFAULT_MPS)
    parser.add_argument("--t4-dir", type=Path, default=_DEFAULT_T4)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--write", action="store_true",
                        help="save PDFs (default is a dry-run listing)")
    args = parser.parse_args()

    dirs = {"MPS": args.mps_dir, "T4": args.t4_dir}
    targets = ["rq1-speedup.pdf", "rq2-hybrid-ladder.pdf"]

    if not args.write:
        print("dry run (pass --write to save):")
        for name in targets:
            print(f"  would write {args.out_dir / name}")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    projections = {
        platform: _load(d / "projection.json") for platform, d in dirs.items()
    }
    out = args.out_dir / "rq1-speedup.pdf"
    _plot_rq1(projections, out)
    print(f"wrote {out}")

    ladders = {
        platform: _rq2_ladders(_load(d / "experiment-2.json"))
        for platform, d in dirs.items()
    }
    out = args.out_dir / "rq2-hybrid-ladder.pdf"
    _plot_rq2(ladders, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
