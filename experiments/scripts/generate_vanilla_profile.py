"""Generate the vanilla baseline section-profiling artifact.

This runs the canonical in-place baseline evaluator with section profiling
enabled and writes the artifact consumed by persist_vanilla_profile.py.

Run from the experiments/ root:
    python scripts/generate_vanilla_profile.py \
        --results-dir canonical-results/mps-results \
        --device mps --write

Then persist the compact thesis snapshot:
    python scripts/persist_vanilla_profile.py \
        --results-dir canonical-results/mps-results --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments import device as device_mod  # noqa: E402
from experiments.candidates import baseline  # noqa: E402
from experiments.schemas import DEFAULT_WORKLOADS, GridSpec  # noqa: E402
from experiments.workloads import task_for_workload, workload_metadata  # noqa: E402


_ARTIFACT_NAME = "experiment-a-vanilla-profiling.json"
_SCHEMA_VERSION = "experiment-a-vanilla-profiling-v1"


def _profile_workload(
    workload_name: str,
    *,
    device,
    grid: GridSpec,
    batch_size: int,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    task = task_for_workload(workload_name, sample_count)
    output = baseline.run(
        task,
        grid,
        batch_size=batch_size,
        device=device,
        seed=seed,
        profile_sections=True,
    )
    grid_points = grid.resolution * grid.resolution
    return {
        "candidate": "baseline_original",
        "metadata": {
            "device": device.type,
            "section_timings_available": output.section_timings is not None,
            "workload": workload_name,
            "wrapped": "generic vanilla point-loop semantics",
        },
        "peak_cpu_memory_bytes": output.peak_cpu_memory_bytes,
        "peak_cuda_memory_bytes": output.peak_cuda_memory_bytes,
        "per_point_latency_s": output.total_grid_s / grid_points,
        "section_timings": output.section_timings,
        "status": "completed" if output.error is None else "error",
        "total_grid_s": output.total_grid_s,
        "workload": workload_metadata(workload_name, sample_count),
        "error": output.error,
    }


def _artifact(
    *,
    output_path: Path,
    device_name: str,
    workloads: tuple[str, ...],
    grid: GridSpec,
    batch_size: int,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    device = device_mod.resolve(device_name)
    started = perf_counter()
    cells: dict[str, Any] = {}
    totals: dict[str, float] = {}

    for workload_name in workloads:
        print(f"[vanilla_profile] workload {workload_name}", flush=True)
        cell = _profile_workload(
            workload_name,
            device=device,
            grid=grid,
            batch_size=batch_size,
            sample_count=sample_count,
            seed=seed,
        )
        cells[workload_name] = cell
        if cell["status"] == "completed":
            totals[workload_name] = cell["total_grid_s"]

    duration_s = perf_counter() - started
    record = {
        "completed_workloads": len(totals),
        "duration_s": duration_s,
        "output_path": str(output_path),
        "status": "completed" if len(totals) == len(workloads) else "completed_with_errors",
        "total_grid_s_by_workload": totals,
        "workload_count": len(workloads),
    }
    return {
        "child_stem": "experiment-a-vanilla-profiling",
        "duration_s": duration_s,
        "output_path": str(output_path),
        "record": record,
        "result": {
            "record": record,
            "schema_version": _SCHEMA_VERSION,
            "workload_count": len(workloads),
            "workloads": cells,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--grid-resolution", type=int, default=8)
    parser.add_argument("--grid-scale", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--workload",
        action="append",
        choices=DEFAULT_WORKLOADS,
        help="workload to profile; repeat to select multiple workloads",
    )
    parser.add_argument(
        "--output-name",
        default=_ARTIFACT_NAME,
        help=f"artifact filename, default: {_ARTIFACT_NAME}",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="write the artifact; without this, print the output path only",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing artifact at the target path",
    )
    args = parser.parse_args()

    results_dir: Path = args.results_dir
    output_path = results_dir / args.output_name
    workloads = tuple(args.workload or DEFAULT_WORKLOADS)
    grid = GridSpec(args.grid_resolution, args.grid_scale)

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists; pass --overwrite to replace it")

    if not args.write:
        print(f"dry run: would write {output_path}")
        print("pass --write to run profiling and write the artifact")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    artifact = _artifact(
        output_path=output_path,
        device_name=args.device,
        workloads=workloads,
        grid=grid,
        batch_size=args.batch_size,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    with output_path.open("w") as handle:
        json.dump(artifact, handle, indent=2, sort_keys=True)
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
