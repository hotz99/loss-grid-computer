#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import asdict
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from experiments import (  # noqa: E402
    calibration_cache,
    functional_eval_experiments,
    hybrid_applicability,
    inventory,
    statistical_analysis,
    vanilla_profiling,
)
from src.backends.base import resolve_device  # noqa: E402
from experiments.functional_eval_experiments import (  # noqa: E402
    DEFAULT_FUNCTIONAL_EVAL_WORKLOADS,
)
from experiments.common import (  # noqa: E402
    configured_workloads,
    put_shared_artifact,
    workload_unavailable_reason,
)

ExperimentRunner = Callable[[SimpleNamespace, Path, dict[str, Any]], dict[str, Any]]


# ------------------------------
# Notebook Globals
# ------------------------------

# Edit these in the notebook, then call run_aio_suite().
DEVICE = "auto"
OUTPUT_DIR: str | None = None
FAIL_FAST = False
VERBOSE_EXPERIMENT_LOGS = False

SEED = 1337
SAMPLE_COUNT = 1024
GRID_RESOLUTION = 8
GRID_SCALE = 1.0
GPU_BATCH_SIZE = 64
ATOL = 1e-6
RTOL = 1e-5

MEASURE_WITHOUT_CACHE = False

CALIBRATION_RETRY = 3
MAX_CPU_WORKER_CANDIDATE: int | None = None

RUN_LABEL: str | None = None
MLTASK_WORKLOADS = list(DEFAULT_FUNCTIONAL_EVAL_WORKLOADS)
FUNCTIONAL_EVAL_WORKLOADS: list[str] | None = None
FUNCTIONAL_EVAL_SAMPLE_COUNTS = [1024]
FUNCTIONAL_EVAL_REPEATS = 5
FUNCTIONAL_EVAL_BATCH_SIZE: int | None = None
POINT_CHUNK_SIZES: list[int] = [32, 64]
MAX_MEMORY_FRACTION: float | None = 0.85
INCLUDE_VMAP_REPRODUCTION = False
INCLUDE_FULL_TEST_SET = False


# ------------------------------
# Experiment Registry
# ------------------------------

# Toggle enabled values from the notebook.
EXPERIMENT_REGISTRY: dict[str, dict[str, Any]] = {
    # hardware/software metadata
    "e0_platform_inventory": {
        "enabled": True,
        "title": "Platform Inventory",
        "run": lambda config, output_dir, shared_state: inventory.run(
            config,
            output_dir,
            shared_state,
            platform_summary=_platform_summary,
        ),
    },
    # Platform preflight for torch.func/vmap primitives.
    "functional_eval_api_probe": {
        "enabled": True,
        "title": "Functional Eval API Preflight",
        "run": functional_eval_experiments.run_api_probe,
    },
    # Experiment A step 0: baseline section timing (perturbation construction, parameter binding, forward+loss, result collection).
    "experiment_a_profiling": {
        "enabled": True,
        "title": "Experiment A: Algorithm Profiling (Step 0 — Baseline Section Timing)",
        "run": vanilla_profiling.run,
    },
    # Experiment A steps 1–3: functional_call and vmap redesign candidates.
    "experiment_a_candidates": {
        "enabled": True,
        "title": "Experiment A: Algorithm Redesign Candidates",
        "run": functional_eval_experiments.run_platform_benchmark,
    },
    # Experiment B: throughput-regime scheduler applicability and device affinity.
    "experiment_b_hybrid_applicability": {
        "enabled": True,
        "title": "Experiment B: Hybrid Applicability",
        "run": hybrid_applicability.run,
    },
    # Experiment C: calibration selection and same-family cache amortization.
    "experiment_c_calibration_cache": {
        "enabled": True,
        "title": "Experiment C: Calibration Cache",
        "run": lambda config, output_dir, shared_state: calibration_cache.run(
            config,
            output_dir,
            shared_state,
            variant_paths=[
                str(ROOT / "assets" / f"cifar10-resnet20-{seed}.pkl")
                for seed in [0, 123, 2023, 123456]
            ],
        ),
    },
    # Statistical analysis: synthesize A/B/C results into paper-ready CI tables.
    "statistical_analysis": {
        "enabled": True,
        "title": "Statistical Analysis",
        "run": lambda config, output_dir, shared_state: statistical_analysis.run(
            config,
            output_dir,
            shared_state,
            experiments=shared_state.get("_experiments", {}),
        ),
    },
    # Experiment D placeholder: compose only after isolated wins exist.
    "experiment_d_merged_stack": {
        "enabled": False,
        "title": "Experiment D: Merged Stack",
        "run": "_run_not_implemented",
    },
    # Deferred placeholder: AMR/progressive visualization is outside A-D.
    "progressive_visualization_deferred": {
        "enabled": False,
        "title": "Progressive Visualization",
        "run": "_run_not_implemented",
    },
}


# ------------------------------
# Entry Point
# ------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "__dataclass_fields__"):
        return _json_safe(asdict(value))
    if isinstance(value, SimpleNamespace):
        return _json_safe(vars(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def main() -> dict[str, Any]:
    summary = run_aio_suite()
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))
    return summary


# ------------------------------
# Suite Orchestration
# ------------------------------


def _config_from_globals() -> SimpleNamespace:
    return SimpleNamespace(
        device=DEVICE,
        output_dir=OUTPUT_DIR,
        fail_fast=FAIL_FAST,
        seed=SEED,
        sample_count=SAMPLE_COUNT,
        grid_resolution=GRID_RESOLUTION,
        grid_scale=GRID_SCALE,
        gpu_batch_size=GPU_BATCH_SIZE,
        atol=ATOL,
        rtol=RTOL,
        measure_without_cache=MEASURE_WITHOUT_CACHE,
        verbose_experiment_logs=VERBOSE_EXPERIMENT_LOGS,
        calibration_retry=CALIBRATION_RETRY,
        max_cpu_worker_candidate=MAX_CPU_WORKER_CANDIDATE,
        run_label=RUN_LABEL,
        mltask_workloads=list(MLTASK_WORKLOADS),
        functional_eval_workloads=(
            None
            if FUNCTIONAL_EVAL_WORKLOADS is None
            else list(FUNCTIONAL_EVAL_WORKLOADS)
        ),
        functional_eval_sample_counts=list(FUNCTIONAL_EVAL_SAMPLE_COUNTS),
        functional_eval_repeats=FUNCTIONAL_EVAL_REPEATS,
        functional_eval_batch_size=(
            GPU_BATCH_SIZE
            if FUNCTIONAL_EVAL_BATCH_SIZE is None
            else FUNCTIONAL_EVAL_BATCH_SIZE
        ),
        point_chunk_sizes=list(POINT_CHUNK_SIZES),
        max_memory_fraction=MAX_MEMORY_FRACTION,
        include_vmap_reproduction=INCLUDE_VMAP_REPRODUCTION,
        include_full_test_set=INCLUDE_FULL_TEST_SET,
    )


def _output_dir(raw: str | None, run_label: str | None) -> Path:
    if raw:
        path = Path(raw)
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = (
            Path("outputs")
            / "platform_experiment_suite"
            / f"{_filename_label(run_label)}{timestamp}"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _filename_label(label: str | None) -> str:
    if not label:
        return ""
    safe = "".join(
        char if char.isalnum() or char in ("-", "_") else "-" for char in label
    )
    safe = safe.strip("-_")
    return f"{safe}-" if safe else ""


def _runner(entry: Any) -> ExperimentRunner:
    if callable(entry):
        return entry
    runner = globals().get(entry)
    if not callable(runner):
        raise ValueError(f"registry runner is not callable: {entry}")
    return runner


def _banner(name: str, marker: str) -> None:
    print(f"\n=== {name}: [{marker}] ===")


def _run_quietly(
    runner: ExperimentRunner,
    config: SimpleNamespace,
    output_dir: Path,
    shared_state: dict[str, Any],
) -> dict[str, Any]:
    previous_verbose = os.environ.get("LGC_VERBOSE_EXPERIMENT_LOGS")
    os.environ["LGC_VERBOSE_EXPERIMENT_LOGS"] = (
        "1" if config.verbose_experiment_logs else "0"
    )
    if config.verbose_experiment_logs:
        try:
            return runner(config, output_dir, shared_state)
        finally:
            _restore_verbose_env(previous_verbose)
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner(config, output_dir, shared_state)
    finally:
        _restore_verbose_env(previous_verbose)


def _restore_verbose_env(previous: str | None) -> None:
    if previous is None:
        os.environ.pop("LGC_VERBOSE_EXPERIMENT_LOGS", None)
    else:
        os.environ["LGC_VERBOSE_EXPERIMENT_LOGS"] = previous


def _total_system_memory_bytes() -> int | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (AttributeError, OSError, ValueError):
        return None


def _platform_summary(
    requested_device: str,
    resolved_device: torch.device,
) -> dict[str, Any]:
    gpu: dict[str, Any] | None = None
    if resolved_device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(resolved_device)
        gpu = {
            "name": torch.cuda.get_device_name(resolved_device),
            "device_count": int(torch.cuda.device_count()),
            "current_device": int(torch.cuda.current_device()),
            "capability": list(torch.cuda.get_device_capability(resolved_device)),
            "total_memory_bytes": int(props.total_memory),
            "multi_processor_count": int(props.multi_processor_count),
        }
    elif resolved_device.type == "mps":
        gpu = {
            "name": "Apple MPS",
            "mps_built": bool(torch.backends.mps.is_built()),
        }

    return {
        "requested_device": requested_device,
        "resolved_device": str(resolved_device),
        "host_os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "cpu": {
            "logical_cores": os.cpu_count(),
            "model": platform.processor() or None,
        },
        "memory": {
            "total_system_memory_bytes": _total_system_memory_bytes(),
            "approx_memory_bandwidth": None,
        },
        "gpu": gpu,
        "env_threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }


def _metric_record(
    name: str, payload: dict[str, Any], duration_s: float | None
) -> dict[str, Any]:
    del name
    record = dict(payload.get("record") or {}) if isinstance(payload, dict) else {}
    record.setdefault("status", payload.get("status") if isinstance(payload, dict) else None)
    record["duration_s"] = duration_s
    if isinstance(payload, dict):
        if payload.get("output_path") is not None:
            record.setdefault("output_path", payload.get("output_path"))
        if payload.get("error") is not None:
            record.setdefault("error", payload["error"])
        if payload.get("reason") is not None:
            record.setdefault("reason", payload["reason"])
    return record


def _precompute_shared_nodes(
    config: SimpleNamespace,
    shared_state: dict[str, Any],
) -> None:
    for workload_name in configured_workloads(config):
        if workload_unavailable_reason(workload_name) is not None:
            continue
        outcome = vanilla_profiling.compute_vanilla_full_grid(workload_name, config)
        if outcome is None:
            continue
        key, result, summary = outcome
        put_shared_artifact(shared_state, key, result, summary)


def run_aio_suite(config: SimpleNamespace | None = None) -> dict[str, Any]:
    config = _config_from_globals() if config is None else config
    output_dir = _output_dir(config.output_dir, config.run_label)
    experiments: dict[str, Any] = {}
    records: dict[str, Any] = {}
    shared_state: dict[str, Any] = {}

    _vanilla_full_grid_consumers = {"experiment_a_profiling", "experiment_b_hybrid_applicability"}
    if any(
        bool(EXPERIMENT_REGISTRY.get(name, {}).get("enabled"))
        for name in _vanilla_full_grid_consumers
    ):
        _precompute_shared_nodes(config, shared_state)

    for name, entry in EXPERIMENT_REGISTRY.items():
        if not bool(entry["enabled"]):
            _banner(name, "finish")
            experiments[name] = {
                "status": "disabled",
                "record": {"status": "disabled"},
            }
            records[name] = _metric_record(name, experiments[name], None)
            continue

        runner = _runner(entry["run"])
        _banner(name, "start")
        start = time.perf_counter()
        try:
            experiments[name] = _run_quietly(
                runner,
                config,
                output_dir,
                shared_state,
            )
            shared_state.setdefault("_experiments", {})[name] = experiments[name]
        except Exception as exc:
            experiments[name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
            if config.fail_fast:
                raise
        duration_s = round(time.perf_counter() - start, 6)
        experiments[name]["duration_s"] = duration_s
        records[name] = _metric_record(name, experiments[name], duration_s)
        output_path = _experiment_payload_path(
            output_dir,
            experiments[name].get("child_stem", name),
            config.run_label,
        )
        records[name]["output_path"] = str(output_path)
        experiments[name]["output_path"] = str(output_path)
        experiments[name]["record"] = records[name]
        _write_experiment_payload(
            output_path,
            experiments[name],
        )
        _banner(name, "finish")

    summary = {
        "schema_version": "platform-experiment-suite-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": _platform_summary(config.device, resolve_device(config.device)),
        "config": dict(vars(config)),
        "registry": {
            name: {"enabled": bool(entry["enabled"])}
            for name, entry in EXPERIMENT_REGISTRY.items()
        },
        "shared_state": _shared_state_summary(shared_state),
        "records": records,
        "experiments": experiments,
    }
    summary_path = output_dir / f"{_filename_label(config.run_label)}summary.json"
    summary_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary["output_path"] = str(summary_path)
    return summary


def _shared_state_summary(shared_state: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in _json_safe(shared_state).items()
        if not key.startswith("_")
    }


# ------------------------------
# Shared writers
# ------------------------------


def _experiment_payload_path(
    output_dir: Path,
    stem: str,
    run_label: str | None = None,
) -> Path:
    return output_dir / f"{_filename_label(run_label)}{stem}.json"


def _write_experiment_payload(
    path: Path,
    payload: dict[str, Any],
) -> Path:
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _run_not_implemented(
    config: SimpleNamespace,
    output_dir: Path,
    shared_state: dict[str, Any],
) -> dict[str, Any]:
    del config
    del output_dir
    del shared_state
    return {
        "status": "disabled",
    }


if __name__ == "__main__":
    main()
