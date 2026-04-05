from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, Optional, cast

import torch


@dataclass(frozen=True)
class Measurement:
    total_s: float
    num_points: int

    @property
    def get_points_per_s(self) -> float:
        if self.total_s <= 0:
            return 0.0
        return float(self.num_points) / float(self.total_s)


@dataclass(frozen=True)
class ComparisonRecord:
    baseline_total_s: float
    speedup: float | None
    allclose: bool | None
    rmse: float | None


def synchronize_device(device: torch.device) -> None:
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif (
            device.type == "mps"
            and hasattr(torch, "mps")
            and hasattr(torch.mps, "synchronize")
        ):
            torch.mps.synchronize()
    except Exception:
        print("failed to sync torch device")
        exit()


@dataclass(frozen=True)
class RunRecord:
    experiment_name: str
    measurement: Measurement
    backend: str
    device: Dict[str, Any]
    config: Dict[str, Any]
    comparison: ComparisonRecord | None
    output_dir: str


@dataclass
class ExperimentResult:
    record: RunRecord
    runtime_log: Dict[str, Any]
    surface: Optional[torch.Tensor]


def _write_csv(record: Dict[str, Any], output_dir: Path) -> None:
    csv_path = output_dir / "results.csv"
    fieldnames = list(record.keys())
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def write_experiment_result(result: ExperimentResult) -> None:
    record = result.record

    output_dir = Path(record.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.snapshot.json"
    breakdown_path = output_dir / "runtime_breakdown.json"
    summary_path = output_dir / "summary.json"
    surface_path = output_dir / "loss_surface.pt"

    config_path.write_text(
        json.dumps(record.config, indent=2, sort_keys=True), encoding="utf-8"
    )
    breakdown_path.write_text(
        json.dumps(result.runtime_log, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if result.surface is not None:
        torch.save(result.surface.cpu(), surface_path)

    summary_row = cast(
        Dict[str, Any],
        {
            "device": record.device,
            "num_points": record.measurement.num_points,
            "points_per_s": record.measurement.get_points_per_s,
            "total_s": record.measurement.total_s,
            "speedup": (
                record.comparison.speedup if record.comparison is not None else None
            ),
            "allclose": (
                record.comparison.allclose if record.comparison is not None else None
            ),
            "rmse": record.comparison.rmse if record.comparison is not None else None,
        },
    )
    _write_csv(summary_row, output_dir)
