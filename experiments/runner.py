#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, is_dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import (  # noqa: E402
    exp_1_algorithm,
    exp_2_hybrid,
    exp_3_cache,
    inventory,
    project,
)
from experiments.schemas import (  # noqa: E402
    Experiment1Config,
    Experiment2Config,
    Experiment3Config,
    GridSpec,
)


OUTPUT_DIR: str | None = None
RUN_LABEL: str | None = None
DEVICE = "auto"
FAIL_FAST = True

RUN_PLATFORM_INVENTORY = False
RUN_EXPERIMENT_1 = True
RUN_EXPERIMENT_2 = True
RUN_EXPERIMENT_3 = True
RUN_PROJECTION = True

WORKLOAD_NAMES: tuple[str, ...] | None = None
SEED = 1337
SAMPLE_COUNT = 1024
GRID_RESOLUTION = 8
EXPERIMENT_3_SESSION_GRID_RESOLUTION = 20
GRID_SCALE = 1.0
GPU_BATCH_SIZE = 64
REPEATS = 1
POINT_CHUNK_SIZES = (32, 64)
MAX_MEMORY_FRACTION: float | None = 0.85
INCLUDE_COMPILE_CANDIDATES = True
CALIBRATION_RETRY = 3
MAX_CPU_WORKER_CANDIDATE: int | None = None


def main() -> Path:
    run_dir = _create_run_dir()
    experiment_1_config = replace(
        Experiment1Config(device=DEVICE),  # type: ignore[arg-type]
        workload_names=_workload_names(),
        seed=SEED,
        sample_count=SAMPLE_COUNT,
        grid=GridSpec(GRID_RESOLUTION, GRID_SCALE),
        batch_size=GPU_BATCH_SIZE,
        repeats=REPEATS,
        point_chunk_sizes=tuple(POINT_CHUNK_SIZES),
        max_memory_fraction=MAX_MEMORY_FRACTION,
        include_compile_candidates=INCLUDE_COMPILE_CANDIDATES,
    )
    experiment_2_config = replace(
        Experiment2Config(device=DEVICE),  # type: ignore[arg-type]
        workload_names=_workload_names(),
        seed=SEED,
        sample_count=SAMPLE_COUNT,
        grid=GridSpec(GRID_RESOLUTION, GRID_SCALE),
        gpu_batch_size=GPU_BATCH_SIZE,
        repeats=REPEATS,
        calibration_retry=CALIBRATION_RETRY,
        max_cpu_worker_candidate=MAX_CPU_WORKER_CANDIDATE,
    )
    experiment_3_config = replace(
        Experiment3Config(device=DEVICE),  # type: ignore[arg-type]
        seed=SEED,
        sample_count=SAMPLE_COUNT,
        session_grid=GridSpec(EXPERIMENT_3_SESSION_GRID_RESOLUTION, GRID_SCALE),
        gpu_batch_size=GPU_BATCH_SIZE,
        calibration_retry=CALIBRATION_RETRY,
        max_cpu_worker_candidate=MAX_CPU_WORKER_CANDIDATE,
    )

    config_payload = {
        "device": DEVICE,
        "run_toggles": {
            "platform_inventory": RUN_PLATFORM_INVENTORY,
            "experiment_1": RUN_EXPERIMENT_1,
            "experiment_2": RUN_EXPERIMENT_2,
            "experiment_3": RUN_EXPERIMENT_3,
            "projection": RUN_PROJECTION,
        },
        "experiment_1": experiment_1_config,
        "experiment_2": experiment_2_config,
        "experiment_3": experiment_3_config,
    }
    _write_json(run_dir / "config.json", config_payload)

    platform = _maybe_run_step(
        name="platform_inventory",
        enabled=RUN_PLATFORM_INVENTORY,
        path=run_dir / "platform.json",
        disabled_payload={"status": "disabled"},
        fn=lambda: inventory.run(DEVICE),
    )
    experiment_1 = _maybe_run_step(
        name="experiment_1",
        enabled=RUN_EXPERIMENT_1,
        path=run_dir / "experiment-1.json",
        disabled_payload={"status": "disabled", "record": {"status": "disabled"}},
        fn=lambda: exp_1_algorithm.run(experiment_1_config),
    )
    experiment_2 = _maybe_run_step(
        name="experiment_2",
        enabled=RUN_EXPERIMENT_2,
        path=run_dir / "experiment-2.json",
        disabled_payload={"status": "disabled", "record": {"status": "disabled"}},
        fn=lambda: exp_2_hybrid.run(experiment_2_config),
    )
    experiment_3 = _maybe_run_step(
        name="experiment_3",
        enabled=(
            RUN_EXPERIMENT_3
            and not isinstance(experiment_1, dict)
            and not isinstance(experiment_2, dict)
        ),
        path=run_dir / "experiment-3.json",
        disabled_payload=_dependency_payload(
            RUN_EXPERIMENT_3,
            "experiment_1, experiment_2",
        ),
        fn=lambda: exp_3_cache.run(experiment_3_config, experiment_1, experiment_2),
    )
    projection = _maybe_run_step(
        name="projection",
        enabled=(
            RUN_PROJECTION
            and not isinstance(experiment_1, dict)
            and not isinstance(experiment_2, dict)
            and not isinstance(experiment_3, dict)
        ),
        path=run_dir / "projection.json",
        disabled_payload=_projection_disabled_payload(
            experiment_1,
            experiment_2,
            experiment_3,
        ),
        fn=lambda: project.project(experiment_1, experiment_2, experiment_3),
    )

    _write_json(
        run_dir / "suite.json",
        {
            "schema_version": "platform-experiment-suite-v2",
            "status": _suite_status(
                platform,
                experiment_1,
                experiment_2,
                experiment_3,
                projection,
            ),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                "config": "config.json",
                "platform": "platform.json",
                "experiment_1": "experiment-1.json",
                "experiment_2": "experiment-2.json",
                "experiment_3": "experiment-3.json",
                "projection": "projection.json",
            },
            "records": {
                "platform": _record(platform),
                "experiment_1": _record(experiment_1),
                "experiment_2": _record(experiment_2),
                "experiment_3": _record(experiment_3),
                "projection": _record(projection),
            },
        },
    )
    print(run_dir)
    return run_dir


def _run_step(name: str, path: Path, fn):
    _banner(name, "start")
    try:
        result = fn()
    except Exception as exc:
        result = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        _write_json(path, result)
        _banner(name, result.get("status", "error"))
        if FAIL_FAST:
            raise
        return result
    _write_json(path, result)
    _banner(name, _step_status(result))
    return result


def _maybe_run_step(
    *,
    name: str,
    enabled: bool,
    path: Path,
    disabled_payload: dict[str, Any],
    fn,
):
    if enabled:
        return _run_step(name, path, fn)
    _write_json(path, disabled_payload)
    _banner(name, _step_status(disabled_payload))
    return disabled_payload


def _dependency_payload(requested: bool, dependency_name: str) -> dict[str, Any]:
    if requested:
        return {
            "status": "skipped",
            "skip_reason": f"requires enabled {dependency_name}",
            "record": {
                "status": "skipped",
                "skip_reason": f"requires enabled {dependency_name}",
            },
        }
    return {"status": "disabled", "record": {"status": "disabled"}}


def _projection_disabled_payload(
    experiment_1: Any,
    experiment_2: Any,
    experiment_3: Any,
) -> dict[str, Any]:
    if not RUN_PROJECTION:
        return {"status": "disabled"}
    missing = [
        name
        for name, value in (
            ("experiment_1", experiment_1),
            ("experiment_2", experiment_2),
            ("experiment_3", experiment_3),
        )
        if isinstance(value, dict)
    ]
    return {
        "status": "skipped",
        "skip_reason": f"requires completed {', '.join(missing)}",
    }


def _create_run_dir() -> Path:
    if OUTPUT_DIR:
        run_dir = Path(OUTPUT_DIR)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = (
            Path("outputs")
            / "platform_experiment_suite"
            / f"{_filename_label(RUN_LABEL)}{timestamp}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _workload_names() -> tuple[str, ...]:
    if WORKLOAD_NAMES is None:
        return Experiment1Config().workload_names
    return tuple(WORKLOAD_NAMES)


def _filename_label(label: str | None) -> str:
    if not label:
        return ""
    safe = "".join(
        char if char.isalnum() or char in ("-", "_") else "-" for char in label
    )
    safe = safe.strip("-_")
    return f"{safe}-" if safe else ""


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_json_safe(payload), indent=2, sort_keys=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    tmp_path.replace(path)
    _read_json(path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _record(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get("record", value)
    return getattr(value, "record", None)


def _suite_status(*steps: Any) -> str:
    statuses = []
    for step in steps:
        if isinstance(step, dict):
            statuses.append(step.get("status"))
        else:
            statuses.append(getattr(step, "status", None))
    if any(status == "error" for status in statuses):
        return "completed_with_errors"
    if any(status == "unknown_workload" for status in statuses):
        return "completed_with_errors"
    if any(status == "planned" for status in statuses):
        return "planned"
    return "completed"


def _step_status(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("status", "completed"))
    return str(getattr(value, "status", "completed"))


def _banner(name: str, marker: str) -> None:
    print(f"[runner] {name} {marker}", flush=True)


if __name__ == "__main__":
    main()
