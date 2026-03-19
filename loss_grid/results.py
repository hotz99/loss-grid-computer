from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from loss_grid.instrumentation import StageBreakdown


@dataclass
class ExperimentResult:
    experiment_name: str
    output_dir: str
    backend: str
    device: Dict[str, Any]
    rank: int
    world_size: int
    surface: Optional[torch.Tensor]
    stage_breakdown: StageBreakdown
    metrics: Dict[str, Any]
    config: Dict[str, Any]
    environment: Dict[str, Any]
    runtime_log: Dict[str, Any]
    is_root: bool = True

    def summary_record(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "throughput_points_per_s": self.metrics.get("throughput_points_per_s"),
            "perturbation_s": self.stage_breakdown.perturbation_s,
            "transfer_s": self.stage_breakdown.transfer_s,
            "forward_s": self.stage_breakdown.forward_s,
            "gpu_kernel_s": self.stage_breakdown.gpu_kernel_s,
            "total_s": self.stage_breakdown.total_s,
            "overlap_efficiency": self.stage_breakdown.overlap_efficiency,
        }


def _write_csv(record: Dict[str, Any], output_dir: Path) -> None:
    csv_path = output_dir / "results.csv"
    fieldnames = list(record.keys())
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def write_summary_table(records: List[Dict[str, Any]], output_dir: str, output_formats: List[str]) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = output_path / "results.csv"
    fieldnames = list(records[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    if "parquet" in output_formats:
        try:
            import pandas as pd
        except ImportError:
            return
        pd.DataFrame(records).to_parquet(output_path / "results.parquet", index=False)


def _write_parquet(record: Dict[str, Any], output_dir: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        return
    parquet_path = output_dir / "results.parquet"
    frame = pd.DataFrame([record])
    if parquet_path.exists():
        previous = pd.read_parquet(parquet_path)
        frame = pd.concat([previous, frame], ignore_index=True)
    frame.to_parquet(parquet_path, index=False)


def write_experiment_result(result: ExperimentResult) -> None:
    output_dir = Path(result.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "config.snapshot.json"
    env_path = output_dir / "environment.json"
    breakdown_path = output_dir / "runtime_breakdown.json"
    summary_path = output_dir / "summary.json"
    surface_path = output_dir / "loss_surface.pt"

    config_path.write_text(json.dumps(result.config, indent=2, sort_keys=True), encoding="utf-8")
    env_path.write_text(json.dumps(result.environment, indent=2, sort_keys=True), encoding="utf-8")
    breakdown_path.write_text(
        json.dumps(result.runtime_log, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(result.summary_record(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    if result.surface is not None:
        torch.save(result.surface.cpu(), surface_path)

    record = result.summary_record()
    _write_csv(record, output_dir)
    if "parquet" in result.config.get("runtime", {}).get("output_formats", []):
        _write_parquet(record, output_dir)
