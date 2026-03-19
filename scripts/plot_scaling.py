#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def load_rows(path: str):
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot scaling metrics from results.csv")
    parser.add_argument("--input", required=True, help="Path to results.csv")
    parser.add_argument("--output", required=True, help="Path to output PNG")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required for plotting") from error

    rows = load_rows(args.input)
    if not rows:
        raise RuntimeError("No rows found in results file")

    rows.sort(key=lambda row: (row.get("backend", ""), int(float(row.get("world_size", "1")))))
    x_values = list(range(len(rows)))
    throughput = [float(row.get("throughput_points_per_s", 0.0) or 0.0) for row in rows]
    labels = [
        f"{row.get('backend', 'unknown')}-w{int(float(row.get('world_size', '1')))}"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x_values, throughput, color="#2f6db0")
    ax.set_ylabel("Throughput (points/s)")
    ax.set_xlabel("Experiment")
    ax.set_title("Loss-grid throughput by backend and worker count")
    ax.set_xticks(x_values)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.tight_layout()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)


if __name__ == "__main__":
    main()
