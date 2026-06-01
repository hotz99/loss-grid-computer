"""Re-measure cold compile cost for the RQ3 cells without rerunning the grid.

compile_s feeds only three derived scalars (N_star_compile, compile_reuse_label,
session_speedup_vs_vanilla). The session timings (T_vanilla, T_gpu_only) are
independent of it and are already in the result files, so the expensive 20x20
sessions do not need rerunning. This script times the cold torch.compile build
for each cell's rq3_config and recomputes the three scalars offline.

The compile is timed inside a fresh Inductor cache (the production path in
exp_3_cache._measure_compile_cost), so each sample is genuinely cold rather than
a warm cache hit left by exp_1 / exp_2.

Run this on the SAME device the results were collected on (the T4 results need a
T4). Triton codegen time is hardware specific, so timing on CPU or MPS would
produce a compile_s that does not apply to the recorded sessions. The script
refuses to patch when the requested device does not match the cells' platform
unless --force-device is given.

Usage (on the Colab T4 runtime):
    python -m experiments.scripts.remeasure_compile_cost \
        --results-dir experiments/results/t4-results \
        --repeats 5 --device cuda --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from experiments import device as device_mod
from experiments.exp_3_cache import (
    _compile_reuse_label,
    _measure_compile_cost,
    _n_star_compile,
    _n_star_json,
    _parse_gpu_candidate,
    _probe_grid_for,
    _session_speedup,
)
from experiments.stats import geometric_mean, t_interval_95
from experiments.workloads import task_for_workload


def _load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _cells(projection: dict[str, Any]) -> list[dict[str, Any]]:
    return projection.get("rq3", {}).get("cells", [])


def _measure_workload(
    cell: dict[str, Any],
    *,
    device,
    batch_size: int,
    seed: int,
    sample_count: int,
    repeats: int,
) -> dict[str, Any]:
    config_name = cell["rq3_config"]
    candidate = _parse_gpu_candidate(config_name)
    task = task_for_workload(cell["workload"], sample_count=sample_count)
    chunk_k = int(cell.get("point_chunk_size_K") or 1)
    # Cold start is grid-size independent (it is the graph build for the chunk
    # shape), so a minimal probe grid that holds one full chunk reproduces it.
    grid = _probe_grid_for(p_max=0, point_chunk_size_k=chunk_k)
    samples = [
        _measure_compile_cost(
            candidate, task, grid,
            gpu_batch_size=batch_size, device=device, seed=seed,
        )
        for _ in range(repeats)
    ]
    interval = t_interval_95(samples)
    return {
        "samples_s": samples,
        "mean_s": geometric_mean(samples),
        "ci_low": interval[0] if interval else None,
        "ci_high": interval[1] if interval else None,
        "repeats": repeats,
    }


def _recompute(cell: dict[str, Any], compile_s: float) -> dict[str, Any]:
    """Apply the production formulas with the new compile_s. Returns the changed
    fields only, leaving the session timings untouched."""
    t_vanilla = cell["T_vanilla"]
    t_gpu_only = cell["T_gpu_only"]
    compiles = bool(cell.get("rq3_config_compiles"))
    n_star = _n_star_compile(compile_s, t_vanilla, t_gpu_only, compiles)
    # session_speedup is suppressed only on an invalid surface; these cells all
    # passed (composition_verdict != surface_invalid), so valid=True holds.
    valid = cell.get("composition_verdict") != "surface_invalid"
    return {
        "compile_s": compile_s,
        "N_star_compile": _n_star_json(n_star),
        "compile_reuse_label": _compile_reuse_label(n_star, compiles),
        "session_speedup_vs_vanilla": _session_speedup(
            t_vanilla, t_gpu_only, compile_s, valid=valid,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--force-device", action="store_true",
        help="time on a device that differs from the cells' recorded platform",
    )
    parser.add_argument(
        "--write", action="store_true",
        help="patch projection.json in place (a .bak copy is kept)",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    projection_path = results_dir / "projection.json"
    config = _load(results_dir / "config.json")
    projection = _load(projection_path)
    cells = _cells(projection)
    if not cells:
        raise SystemExit(f"no rq3 cells found in {projection_path}")

    device = device_mod.resolve(args.device)
    exp3_cfg = config.get("experiment_3", {})
    batch_size = int(exp3_cfg.get("gpu_batch_size", 64))
    seed = int(exp3_cfg.get("seed", 1337))
    sample_count = int(exp3_cfg.get("sample_count", 1024))

    recorded_platforms = {c.get("platform") for c in cells}
    if device.type not in recorded_platforms and not args.force_device:
        raise SystemExit(
            f"device {device.type!r} does not match recorded platform(s) "
            f"{recorded_platforms}. Compile time is hardware specific; run this "
            f"on the matching device or pass --force-device to override."
        )

    sidecar: dict[str, Any] = {
        "device": device.type,
        "repeats": args.repeats,
        "batch_size": batch_size,
        "seed": seed,
        "workloads": {},
    }

    print(
        f"{'workload':<34} {'config':<22} "
        f"{'compile_s old->new':<22} {'N* o->n':<10} {'sess_speedup o->n'}"
    )
    for cell in cells:
        if not cell.get("rq3_config_compiles"):
            continue
        measurement = _measure_workload(
            cell, device=device, batch_size=batch_size, seed=seed,
            sample_count=sample_count, repeats=args.repeats,
        )
        new_compile_s = measurement["mean_s"] or 0.0
        old = {
            "compile_s": cell.get("compile_s"),
            "N_star_compile": cell.get("N_star_compile"),
            "session_speedup_vs_vanilla": cell.get("session_speedup_vs_vanilla"),
        }
        updated = _recompute(cell, new_compile_s)
        sidecar["workloads"][cell["workload"]] = {
            "rq3_config": cell["rq3_config"],
            "measurement": measurement,
            "old": old,
            "new": updated,
        }
        print(
            f"{cell['workload']:<34} {cell['rq3_config']:<22} "
            f"{old['compile_s']:.2f}->{new_compile_s:.2f}{'':<10} "
            f"{str(old['N_star_compile']):>3}->{str(updated['N_star_compile']):<4} "
            f"{old['session_speedup_vs_vanilla']:.2f}->"
            f"{updated['session_speedup_vs_vanilla']:.2f}"
        )
        if args.write:
            cell.update(updated)

    sidecar_path = results_dir / "compile_cost_remeasured.json"
    with sidecar_path.open("w") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
    print(f"\nwrote raw samples + before/after to {sidecar_path}")

    if args.write:
        backup = projection_path.with_suffix(".json.bak")
        backup.write_text(json.dumps(_load(projection_path), indent=2))
        with projection_path.open("w") as handle:
            json.dump(projection, handle, indent=2, sort_keys=True)
        print(f"patched {projection_path} (backup at {backup})")
    else:
        print("dry run: pass --write to patch projection.json")


if __name__ == "__main__":
    main()
