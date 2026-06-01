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
    CandidateRunResult,
    Experiment1Config,
    Experiment1Result,
    Experiment2Config,
    Experiment2Result,
    Experiment3Config,
    GridSpec,
)
from experiments.surface_gate import validate_surface  # noqa: E402


class PipelineContractError(RuntimeError):
    """An upstream result does not match the shape a downstream experiment
    consumes. Raised before the downstream step runs so a malformed producer
    fails fast instead of wasting downstream runtime or emitting an invalid
    RQ3 verdict."""


OUTPUT_DIR: str | None = None
RUN_LABEL: str | None = None
DEVICE = "auto"
FAIL_FAST = True

RUN_PLATFORM_INVENTORY = True
RUN_EXPERIMENT_1 = True
RUN_EXPERIMENT_2 = True
RUN_EXPERIMENT_3 = True
RUN_PROJECTION = True

# Optional reuse sources for posthoc runs. Set these to completed upstream
# artifact paths and disable the matching experiment toggle to run exp3 or
# projection against frozen experiment-1/experiment-2 records.
src = "./outputs/platform_experiment_suite/20260531T120708Z"
REUSE_EXPERIMENT_1_FROM: str | None = None  # f"{src}/experiment-1.json"
REUSE_EXPERIMENT_2_FROM: str | None = None  # f"{src}/experiment-2.json"

WORKLOAD_NAMES: tuple[str, ...] | None = None
SEED = 1337
SAMPLE_COUNT = 1024
GRID_RESOLUTION = 8
EXPERIMENT_3_SESSION_GRID_RESOLUTION = 20
GRID_SCALE = 1.0
GPU_BATCH_SIZE = 64
# RQ2 slowdown ladder ceiling (base-2 rungs from slow=1 up to this value)
SLOWDOWN_CEILING = 16
# R = 1 breaks exp2->exp3 piping, since no CIs resolved
REPEATS = 5
POINT_CHUNK_SIZES = (32, 64)
MAX_MEMORY_FRACTION: float | None = 0.85
INCLUDE_COMPILE_CANDIDATES = True
COMPOSITION_SELECTION_PATIENCE = 3
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
        slowdown_ceiling=SLOWDOWN_CEILING,
        max_cpu_worker_candidate=MAX_CPU_WORKER_CANDIDATE,
    )
    experiment_3_config = replace(
        Experiment3Config(device=DEVICE),  # type: ignore[arg-type]
        seed=SEED,
        sample_count=SAMPLE_COUNT,
        session_grid=GridSpec(EXPERIMENT_3_SESSION_GRID_RESOLUTION, GRID_SCALE),
        gpu_batch_size=GPU_BATCH_SIZE,
        composition_selection_patience=COMPOSITION_SELECTION_PATIENCE,
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
        "reuse_sources": {
            "experiment_1": REUSE_EXPERIMENT_1_FROM,
            "experiment_2": REUSE_EXPERIMENT_2_FROM,
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
    if not RUN_EXPERIMENT_1 and REUSE_EXPERIMENT_1_FROM:
        experiment_1 = _reuse_experiment_1(
            Path(REUSE_EXPERIMENT_1_FROM),
            experiment_1_config,
            run_dir / "experiment-1.json",
        )
    else:
        experiment_1 = _maybe_run_step(
            name="experiment_1",
            enabled=RUN_EXPERIMENT_1,
            path=run_dir / "experiment-1.json",
            disabled_payload={"status": "disabled", "record": {"status": "disabled"}},
            fn=lambda: exp_1_algorithm.run(experiment_1_config),
        )
    if not RUN_EXPERIMENT_2 and REUSE_EXPERIMENT_2_FROM:
        experiment_2 = _reuse_experiment_2(
            Path(REUSE_EXPERIMENT_2_FROM),
            experiment_2_config,
            run_dir / "experiment-2.json",
        )
    else:
        experiment_2 = _maybe_run_step(
            name="experiment_2",
            enabled=RUN_EXPERIMENT_2,
            path=run_dir / "experiment-2.json",
            disabled_payload={"status": "disabled", "record": {"status": "disabled"}},
            fn=lambda: exp_2_hybrid.run(experiment_2_config),
        )
    experiment_3_inputs_ready = (
        RUN_EXPERIMENT_3
        and not isinstance(experiment_1, dict)
        and not isinstance(experiment_2, dict)
    )
    if experiment_3_inputs_ready:
        _assert_experiment_3_inputs(experiment_1, experiment_2)
    experiment_3 = _maybe_run_step(
        name="experiment_3",
        enabled=experiment_3_inputs_ready,
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
        fn=lambda: project.project(
            _read_json(run_dir / "experiment-1.json"),
            _read_json(run_dir / "experiment-2.json"),
            _read_json(run_dir / "experiment-3.json"),
        ),
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
        _assert_output_surfaces(name, result)
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


def _reuse_experiment_1(
    source: Path, config: Experiment1Config, dest: Path
) -> Experiment1Result:
    """Load a completed experiment-1.json so exp3 can consume a frozen RQ1
    winner without rerunning the algorithm sweep."""
    payload = _read_json(source)
    if payload.get("status") != "completed":
        raise SystemExit(
            f"reuse exp1: {source} has status {payload.get('status')!r}, expected 'completed'"
        )
    runs = tuple(
        CandidateRunResult(
            workload_name=item["workload_name"],
            candidate=item["candidate"],
            role=item["role"],
            repeat=int(item["repeat"]),
            status=item["status"],
            trial_order=tuple(item.get("trial_order") or ()),
            total_grid_s=item.get("total_grid_s"),
            records=tuple(tuple(record) for record in (item.get("records") or ())),
            validation=item.get("validation"),
            diagnostics=item.get("diagnostics") or {},
            error=item.get("error"),
        )
        for item in payload.get("runs", ())
    )
    result = Experiment1Result(
        status=payload["status"],
        schema_version=payload["schema_version"],
        config=config,
        trials=tuple(payload.get("trials", ())),
        runs=runs,
        aggregates=tuple(payload.get("aggregates", ())),
        rq3_config=payload.get("rq3_config")
        or (payload.get("record") or {}).get("rq3_config")
        or "baseline",
        composition=payload.get("composition", {}),
        record=payload.get("record", {}),
    )
    _assert_output_surfaces("experiment_1", result)
    _write_json(dest, result)
    _banner("experiment_1", f"reused:{_step_status(payload)}")
    return result


def _reuse_experiment_2(
    source: Path, config: Experiment2Config, dest: Path
) -> Experiment2Result:
    """Load a completed experiment-2.json and rebuild its result so exp3 can
    consume r_native per workload without recomputing exp2. The artifact is
    copied into the run directory so projection and the suite manifest read a
    consistent exp2."""
    payload = _read_json(source)
    if payload.get("status") != "completed":
        raise SystemExit(
            f"reuse exp2: {source} has status {payload.get('status')!r}, expected 'completed'"
        )
    result = Experiment2Result(
        status=payload["status"],
        schema_version=payload["schema_version"],
        config=config,
        result=payload.get("result", {}),
        record=payload.get("record", {}),
    )
    _assert_output_surfaces("experiment_2", result)
    _write_json(dest, result)
    _banner("experiment_2", f"reused:{_step_status(payload)}")
    return result


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PipelineContractError(message)


def _assert_output_surfaces(name: str, result: Any) -> None:
    if name not in {"experiment_1", "experiment_2", "experiment_3"}:
        return
    if _step_status(result) != "completed":
        return

    config = getattr(result, "config", None)
    surface_gate = getattr(config, "surface_gate", None)
    _require(
        surface_gate is not None,
        f"{name} completed but exposes no surface gate config",
    )

    pairs = list(_surface_pairs(name, result))
    _require(
        bool(pairs) or name == "experiment_1",
        f"{name} completed but emitted no runner-verifiable surface pairs",
    )
    validations = []
    for pair in pairs:
        label = _surface_pair_label(name, pair)
        baseline = _surface_records(pair, "baseline_records", label)
        candidate = _surface_records(pair, "candidate_records", label)
        try:
            validation = validate_surface(candidate, baseline, surface_gate)
        except AssertionError as exc:
            raise PipelineContractError(
                f"{label} failed surface gate shape check: {exc}"
            ) from exc
        if not validation["valid"]:
            raise PipelineContractError(
                f"{label} failed surface gate: "
                f"mismatches={validation['mismatch_count']} "
                f"max_abs_error={validation['max_abs_error']} "
                f"rtol={surface_gate.rel_tol} atol={surface_gate.abs_tol}"
            )
        validations.append({"label": label, **validation})

    if validations:
        _attach_runner_surface_validation(result, validations)


def _surface_pairs(name: str, result: Any):
    if name == "experiment_1":
        yield from _experiment_1_surface_pairs(result)
    elif name == "experiment_2":
        yield from _experiment_2_surface_pairs(result)
    elif name == "experiment_3":
        yield from _experiment_3_surface_pairs(result)


def _experiment_1_surface_pairs(result: Experiment1Result):
    baseline_by_key: dict[tuple[str, int], CandidateRunResult] = {}
    for run in result.runs:
        if run.status == "ok" and run.role == "baseline":
            baseline_by_key[(run.workload_name, run.repeat)] = run

    for run in result.runs:
        if run.status != "ok" or run.role == "baseline":
            continue
        key = (run.workload_name, run.repeat)
        baseline = baseline_by_key.get(key)
        _require(
            baseline is not None,
            "experiment_1 completed with candidate grid "
            f"{run.candidate!r} repeat={run.repeat} but no baseline grid",
        )
        yield {
            "workload": run.workload_name,
            "candidate": run.candidate,
            "baseline": baseline.candidate,
            "repeat": run.repeat,
            "baseline_records": baseline.records,
            "candidate_records": run.records,
        }


def _experiment_2_surface_pairs(result: Experiment2Result):
    workloads = _experiment_workloads(result)
    for workload_name, workload in workloads.items():
        if not isinstance(workload, dict) or workload.get("status") != "completed":
            continue
        ladder = workload.get("ladder")
        _require(
            isinstance(ladder, list) and bool(ladder),
            f"experiment_2 workload {workload_name!r} has no ladder surface pairs",
        )
        for rung_index, rung in enumerate(ladder):
            _require(
                isinstance(rung, dict),
                f"experiment_2 workload {workload_name!r} ladder[{rung_index}] is not a dict",
            )
            pairs = rung.get("surface_pairs")
            _require(
                isinstance(pairs, list) and bool(pairs),
                "experiment_2 completed rung missing runner-verifiable surface pairs: "
                f"workload={workload_name!r} rung={rung_index}",
            )
            for pair in pairs:
                _require(
                    isinstance(pair, dict),
                    "experiment_2 emitted malformed surface pair: "
                    f"workload={workload_name!r} rung={rung_index}",
                )
                yield {
                    "workload": workload_name,
                    "slowdown_factor": rung.get("slowdown_factor"),
                    **pair,
                }


def _experiment_3_surface_pairs(result: Experiment3Result):
    pairs = (
        result.result.get("surface_pairs") if isinstance(result.result, dict) else None
    )
    _require(
        isinstance(pairs, list) and bool(pairs),
        "experiment_3 completed but emitted no runner-verifiable surface pairs",
    )
    for pair in pairs:
        _require(isinstance(pair, dict), "experiment_3 emitted malformed surface pair")
        yield pair


def _experiment_workloads(result: Experiment2Result) -> dict[str, Any]:
    if isinstance(result.result, dict) and isinstance(
        result.result.get("workloads"), dict
    ):
        return result.result["workloads"]
    if isinstance(result.record, dict) and isinstance(
        result.record.get("workloads"), dict
    ):
        return result.record["workloads"]
    raise PipelineContractError("experiment_2 completed but exposes no workloads")


def _surface_records(pair: dict[str, Any], key: str, label: str):
    _require(key in pair, f"{label} missing {key}")
    records = pair[key]
    _require(
        isinstance(records, (list, tuple)),
        f"{label} {key} must be a sequence",
    )
    return records


def _surface_pair_label(name: str, pair: dict[str, Any]) -> str:
    parts = [name]
    for key in (
        "workload",
        "candidate",
        "baseline",
        "repeat",
        "checkpoint_path",
        "slowdown_factor",
    ):
        if pair.get(key) is not None:
            parts.append(f"{key}={pair[key]}")
    return " ".join(parts)


def _attach_runner_surface_validation(
    result: Any,
    validations: list[dict[str, Any]],
) -> None:
    payload = {
        "valid": True,
        "surface_pair_count": len(validations),
        "validations": validations,
    }
    record = getattr(result, "record", None)
    if isinstance(record, dict):
        record["runner_surface_validation"] = payload
    result_payload = getattr(result, "result", None)
    if isinstance(result_payload, dict):
        result_payload["runner_surface_validation"] = payload


def _assert_experiment_3_inputs(
    experiment_1: Experiment1Result,
    experiment_2: Experiment2Result,
) -> None:
    """Validate the exp1/exp2 -> exp3 boundary before exp3 runs. The composition
    sweep runs the full workload set with no affinity filter, so the old
    hybrid-affinity selection gate is gone. exp3 reads exp1's per-workload
    rq3_config map (falling back to exp1.rq3_config) to fix the GPU evaluator per
    cell, and consumes exp2 only for r_native per workload, the native-r
    reference the optional slowdown overlay maps against. The shape checked here
    is the minimum that keeps those two reads well-formed; a reached threshold is
    no longer required, since the sweep does not select on it."""
    _require(
        isinstance(experiment_1.record, dict),
        f"exp1.record must be a dict, got {type(experiment_1.record).__name__}",
    )
    _require(
        isinstance(experiment_1.rq3_config, str) and experiment_1.rq3_config != "",
        "exp1.rq3_config must be a non-empty config name for exp3 to resolve the GPU config",
    )

    record = experiment_2.record
    _require(
        isinstance(record, dict),
        f"exp2.record must be a dict, got {type(record).__name__}",
    )
    workloads = record.get("workloads")
    _require(
        isinstance(workloads, dict) and bool(workloads),
        "exp2.record['workloads'] must be a non-empty dict for RQ3 to read r_native per cell",
    )
    for name, payload in workloads.items():
        _require(
            isinstance(payload, dict),
            f"exp2 workload {name!r} payload must be a dict",
        )
        if payload.get("status") != "completed":
            continue
        predictor = payload.get("regime_predictor")
        _require(
            isinstance(predictor, dict),
            f"exp2 workload {name!r} is completed but exposes no regime_predictor for r_native",
        )


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
