"""Persist a compact baseline section-decomposition snapshot for tab:profile.

The vanilla-profiling artifact (experiment-a-vanilla-profiling) carries the full
runner record: per-workload section_timings plus dataset, memory, and checkpoint
metadata. The thesis baseline-decomposition table only needs four numbers per
(platform, workload) cell: total_grid_s and the perturbation / binding /
batch-eval shares of it. This script reads the artifact and writes those numbers
into a stable snapshot file so the table is sourced from a small, reviewable
sidecar instead of the large runner output.

This reads recorded timings; it does not re-run any grid. Run it once per results
directory after a profiling run lands.

Run from the experiments/ root:
    python scripts/persist_vanilla_profile.py \
        --results-dir canonical-results/t4-results \
        --results-dir canonical-results/mps-results --write
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

_ARTIFACT_GLOB = "*experiment-a-vanilla-profiling.json"
_SNAPSHOT_NAME = "vanilla_profile_snapshot.json"


def _load(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def _find_artifact(results_dir: Path) -> Path:
    matches = sorted(results_dir.glob(_ARTIFACT_GLOB))
    if not matches:
        raise SystemExit(f"no vanilla-profiling artifact in {results_dir}")
    if len(matches) > 1:
        raise SystemExit(
            f"multiple vanilla-profiling artifacts in {results_dir}: {matches}"
        )
    return matches[0]


def _workloads(artifact: dict[str, Any]) -> dict[str, Any]:
    workloads = artifact.get("result", {}).get("workloads")
    if not workloads:
        raise SystemExit("artifact has no result.workloads")
    return workloads


def _share(part: float, total: float) -> float | None:
    if not total:
        return None
    return 100.0 * part / total


def _cell(record: dict[str, Any]) -> dict[str, Any]:
    timings = record["section_timings"]
    total = timings["total_grid_s"]
    return {
        "device": record.get("metadata", {}).get("device"),
        "total_grid_s": total,
        "perturbation_pct": _share(timings["perturbation_s"], total),
        "binding_pct": _share(timings["binding_s"], total),
        "batch_eval_pct": _share(timings["batch_eval_s"], total),
    }


def _snapshot(results_dir: Path) -> dict[str, Any]:
    artifact = _load(_find_artifact(results_dir))
    workloads = _workloads(artifact)
    cells = {name: _cell(record) for name, record in sorted(workloads.items())}
    return {
        "source_artifact": _find_artifact(results_dir).name,
        "workloads": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, action="append", required=True,
        help="results directory holding a vanilla-profiling artifact "
        "(repeatable)",
    )
    parser.add_argument(
        "--write", action="store_true",
        help=f"write {_SNAPSHOT_NAME} into each results dir",
    )
    args = parser.parse_args()

    header = (
        f"{'workload':<34} {'device':<6} {'total_s':>9} "
        f"{'perturb%':>9} {'binding%':>9} {'batch%':>9}"
    )
    for results_dir in args.results_dir:
        snapshot = _snapshot(results_dir)
        print(f"\n{results_dir}  ({snapshot['source_artifact']})")
        print(header)
        for name, cell in snapshot["workloads"].items():
            print(
                f"{name:<34} {str(cell['device']):<6} "
                f"{cell['total_grid_s']:>9.3f} "
                f"{cell['perturbation_pct']:>9.2f} "
                f"{cell['binding_pct']:>9.2f} "
                f"{cell['batch_eval_pct']:>9.2f}"
            )
        if args.write:
            out = results_dir / _SNAPSHOT_NAME
            with out.open("w") as handle:
                json.dump(snapshot, handle, indent=2, sort_keys=True)
            print(f"wrote {out}")

    if not args.write:
        print("\ndry run: pass --write to persist snapshots")


if __name__ == "__main__":
    main()
