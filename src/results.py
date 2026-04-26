from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Optional, TypeAlias

import torch

from src.config import (
    ExperimentConfig,
    experiment_config_from_dict,
    experiment_config_to_dict,
)


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
    speedup: Optional[float]
    equal: Optional[bool]
    rmse: Optional[float]


@dataclass(frozen=True)
class DeviceRecord:
    gpu: str
    cpu: int


def synchronize_device(device: torch.device):
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
    device: DeviceRecord
    config: ExperimentConfig
    comparison: Optional[ComparisonRecord]
    output_dir: str


@dataclass
class ExperimentResult:
    record: RunRecord
    runtime_log: dict[str, Any]
    records: list[tuple[int, int, float]]


SUMMARY_FILENAME = "summary.json"
CONFIG_SNAPSHOT_FILENAME = "config.snapshot.json"
SURFACE_FILENAME = "loss_surface.json"
SurfacePoint: TypeAlias = tuple[int, int, float]
Surface: TypeAlias = list[SurfacePoint]


def load_json(path: str | Path) -> Any:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_measurement(raw: Any) -> Measurement:
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid measurement payload: {raw!r}")
    return Measurement(
        total_s=float(raw["total_s"]),
        num_points=int(raw["num_points"]),
    )


def _parse_device_record(raw: Any) -> DeviceRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid device payload: {raw!r}")
    return DeviceRecord(
        gpu=str(raw["gpu"]),
        cpu=int(raw["cpu"]),
    )


def _parse_comparison_record(raw: Any) -> Optional[ComparisonRecord]:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid comparison payload: {raw!r}")
    return ComparisonRecord(
        baseline_total_s=float(raw["baseline_total_s"]),
        speedup=None if raw.get("speedup") is None else float(raw["speedup"]),
        equal=None if raw.get("equal") is None else bool(raw["equal"]),
        rmse=None if raw.get("rmse") is None else float(raw["rmse"]),
    )


def _parse_run_record(raw: Any) -> RunRecord:
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid run summary payload: {raw!r}")
    return RunRecord(
        experiment_name=str(raw["experiment_name"]),
        measurement=_parse_measurement(raw["measurement"]),
        backend=str(raw["backend"]),
        device=_parse_device_record(raw["device"]),
        config=experiment_config_from_dict(raw["config"]),
        comparison=_parse_comparison_record(raw.get("comparison")),
        output_dir=str(raw["output_dir"]),
    )


def load_run_summary(run_dir: str | Path) -> RunRecord:
    return _parse_run_record(load_json(Path(run_dir) / SUMMARY_FILENAME))


def load_surface(run_dir: str | Path) -> Surface:
    surface_path = Path(run_dir) / SURFACE_FILENAME
    if not surface_path.exists():
        raise FileNotFoundError(f"Missing surface file: {surface_path}")
    return [
        (int(record[0]), int(record[1]), float(record[2]))
        for record in load_json(surface_path)
    ]


def find_latest_matching_run_dir(
    output_root: str,
    experiment_name: str,
    expected_config: dict[str, Any],
):
    output_root_path = Path(output_root)
    prefix = f"{experiment_name}-"
    matches = [
        path
        for path in output_root_path.iterdir()
        if path.is_dir()
        and path.name.startswith(prefix)
        and (path / SUMMARY_FILENAME).exists()
        and (path / CONFIG_SNAPSHOT_FILENAME).exists()
        and load_json(path / CONFIG_SNAPSHOT_FILENAME) == expected_config
    ]
    if not matches:
        raise FileNotFoundError(
            "No cached baseline run matches the current baseline config under "
            f"{output_root_path} with prefix {prefix!r}"
        )
    matches.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0]


def find_cached_run_dir(config: ExperimentConfig) -> Path:
    return find_latest_matching_run_dir(
        config.runtime.output_root,
        config.experiment_name,
        experiment_config_to_dict(config),
    )


def load_cached_run_summary(config: ExperimentConfig) -> RunRecord:
    return load_run_summary(find_cached_run_dir(config))


def load_cached_run_with_surface(config: ExperimentConfig) -> tuple[str, RunRecord]:
    run_dir = find_cached_run_dir(config)
    load_surface(run_dir)
    return str(run_dir), load_run_summary(run_dir)


def _round_floats(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 3)
    if isinstance(value, dict):
        return {key: _round_floats(inner_value) for key, inner_value in value.items()}
    if isinstance(value, list):
        return [_round_floats(inner_value) for inner_value in value]
    if isinstance(value, tuple):
        return tuple(_round_floats(inner_value) for inner_value in value)
    return value


def to_pretty_json(payload: Any) -> str:
    return json.dumps(_round_floats(payload), indent=2, sort_keys=True)


def write_json(path: Path, payload: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        to_pretty_json(payload),
        encoding="utf-8",
    )


def write_summary_json(
    config: ExperimentConfig,
    filename: str,
    payload: Any,
):
    summary_dir = Path(config.runtime.output_root) / f"{config.experiment_name}-summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    path = summary_dir / filename
    write_json(path, payload)
    return str(path)


def write_experiment_result(result: ExperimentResult):
    record = result.record

    output_dir = Path(record.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / CONFIG_SNAPSHOT_FILENAME
    breakdown_path = output_dir / "runtime_breakdown.json"
    summary_path = output_dir / SUMMARY_FILENAME
    surface_path = output_dir / SURFACE_FILENAME

    write_json(config_path, experiment_config_to_dict(record.config))
    write_json(breakdown_path, result.runtime_log)
    write_json(
        summary_path,
        {
            **asdict(record),
            "config": experiment_config_to_dict(record.config),
        },
    )

    if result.records is not None:
        write_json(surface_path, result.records)
