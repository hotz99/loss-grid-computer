from __future__ import annotations

import importlib
import inspect
import json
import math
import statistics
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch

from src.backends.vanilla import run as run_vanilla_backend
from src.functional_eval.memory import SectionTimings
from src.functional_eval.validation import compare_surfaces
from src.schemas import GridSpec, SchedulerRequest, VanillaMode
from src.workloads import WORKLOADS


Surface = list[tuple[int, int, float]]
Runner = Callable[..., Any]
DEFAULT_WORKLOAD_NAME = "cifar10_resnet20_classification"


@dataclass(frozen=True)
class FunctionalEvalConfig:
    request: SchedulerRequest
    seed: int = 1337
    point_chunk_sizes: tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    rel_tol: float = 1e-5
    abs_tol: float = 1e-6
    max_memory_fraction: float | None = 0.85
    run_label: str | None = None


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    runner: Runner | None
    kwargs: dict[str, Any]
    components: tuple[str, ...]
    applies_to_sections: tuple[str, ...]
    hypothesis: str
    skipped_reason: str | None = None


@dataclass(frozen=True)
class CandidateRun:
    candidate: str
    repeat: int
    status: str
    records: Surface
    timings: SectionTimings
    total_grid_s: float | None
    per_point_latency_s: float | None
    per_batch_eval_s: float | None
    peak_cpu_memory_bytes: int | None
    peak_cuda_memory_bytes: int | None
    validation: dict[str, Any] | None
    speedup_vs_baseline: float | None
    metadata: dict[str, Any]
    error: str | None = None


def build_default_request(
    *,
    workload_name: str = DEFAULT_WORKLOAD_NAME,
    device: str = "auto",
    sample_count: int | None = None,
    batch_size: int = 32,
    resolution: int = 8,
    scale: float = 1.0,
) -> SchedulerRequest:
    try:
        definition = WORKLOADS[workload_name]
    except KeyError as exc:
        available = ", ".join(sorted(WORKLOADS))
        raise ValueError(
            f"unknown functional-eval workload_name {workload_name!r}; "
            f"available workloads: {available}"
        ) from exc
    task = definition.spec
    if sample_count is not None:
        task = replace(task, dataset=replace(task.dataset, sample_count=sample_count))
    return SchedulerRequest(
        task=task,
        grid=GridSpec(resolution=resolution, scale=scale),
        mode=VanillaMode(gpu_batch_size=batch_size),
        device=device,  # type: ignore[arg-type]
    )


def run_experiment(config: FunctionalEvalConfig, repeat: int = 0) -> dict[str, Any]:
    """Execute a single measured run of all candidates.

    Warm-up and repeat orchestration belong at the caller level (runner.py /
    run_platform_suite).  This function runs every candidate exactly once and
    returns its raw results.  The caller is responsible for discarding warm-up
    runs and aggregating across repetitions.
    """
    device = _resolve_device(config.request.device)
    candidate_specs = _candidate_specs(config)

    baseline_spec = CandidateSpec(
        name="baseline_original",
        runner=_run_profiled_vanilla_baseline,
        kwargs={},
        components=("original_in_place_mutation",),
        applies_to_sections=(
            "perturbation_construction",
            "parameter_binding",
            "batch_forward_loss",
        ),
        hypothesis="reference behavior and runtime target",
    )

    baseline = _run_one_candidate(
        baseline_spec,
        config,
        repeat,
        baseline_records=None,
        baseline_time_s=None,
    )
    all_runs: list[CandidateRun] = [baseline]

    for spec in candidate_specs:
        all_runs.append(
            _run_one_candidate(
                spec,
                config,
                repeat,
                baseline_records=baseline.records,
                baseline_time_s=baseline.total_grid_s,
            )
        )

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "platform": _platform_metadata(device),
        "config": _request_summary(config),
        "candidate_summary": _summarize_runs(all_runs),
        "runs": [_json_safe(asdict(run)) for run in all_runs],
    }
    return summary


def _run_profiled_vanilla_baseline(
    request: SchedulerRequest,
    *,
    seed: int = 1337,
) -> dict[str, Any]:
    result = run_vanilla_backend(request, seed=seed, profile_sections=True)
    section_timings = dict(result.runtime_log.get("section_timings") or {})
    section_timings["total_grid_s"] = result.record.measurement.total_s
    return {
        "candidate": "baseline_original",
        "records": result.records,
        "timings": SectionTimings(**section_timings),
        "peak_cpu_memory_bytes": None,
        "peak_cuda_memory_bytes": None,
        "metadata": {
            "wrapped": "canonical vanilla backend",
            "workload": request.task.name,
            "device": result.record.device.gpu,
            "section_timings_available": True,
            "backend": result.record.backend,
        },
    }


def _candidate_specs(config: FunctionalEvalConfig) -> list[CandidateSpec]:
    specs: list[CandidateSpec] = []
    sequential_runner, sequential_skip = _load_runner(
        "src.functional_eval.sequential",
        (
            "run_functional_sequential_surface",
            "run_functional_sequential",
            "run_sequential",
            "run",
            "evaluate",
        ),
    )
    specs.append(
        CandidateSpec(
            name="functional_sequential",
            runner=sequential_runner,
            kwargs={},
            components=("functional_call",),
            applies_to_sections=("parameter_binding",),
            hypothesis=(
                "replace in-place parameter mutation with explicit functional "
                "parameter binding while preserving the original point and "
                "batch loops; this does not vectorize forward/loss work"
            ),
            skipped_reason=sequential_skip,
        )
    )

    vmapped_runner, vmapped_skip = _load_runner(
        "src.functional_eval.vmapped",
        (
            "run_vmapped_surface",
            "run_vmapped",
            "run_functional_vmapped",
            "run",
            "evaluate",
        ),
    )
    for chunk_size in config.point_chunk_sizes:
        specs.append(
            CandidateSpec(
                name=f"vmapped_chunk_{chunk_size}",
                runner=vmapped_runner,
                kwargs={"point_chunk_size": chunk_size},
                components=("functional_call", "vmap", "point_chunking"),
                applies_to_sections=(
                    "perturbation_construction",
                    "parameter_binding",
                    "batch_forward_loss",
                ),
                hypothesis=(
                    "use vmap over functional evaluation to evaluate multiple "
                    "perturbed variants of the original model for the same "
                    "dataset batch, with chunk size as the memory/runtime "
                    "control"
                ),
                skipped_reason=vmapped_skip,
            )
        )
    return specs


def _load_runner(module_name: str, names: tuple[str, ...]) -> tuple[Runner | None, str | None]:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            return None, f"{module_name} is not available yet"
        return None, f"{module_name} import failed: {type(exc).__name__}: {exc}"
    except Exception as exc:
        return None, f"{module_name} import failed: {type(exc).__name__}: {exc}"

    for name in names:
        runner = getattr(module, name, None)
        if callable(runner):
            return runner, None
    return None, f"{module_name} has no runner named one of {names}"


def _run_one_candidate(
    spec: CandidateSpec,
    config: FunctionalEvalConfig,
    repeat: int,
    *,
    baseline_records: Surface | None,
    baseline_time_s: float | None,
) -> CandidateRun:
    if spec.runner is None:
        return _skipped_run(spec, repeat)

    try:
        raw_result = _call_runner(
            spec.runner,
            config.request,
            seed=config.seed,
            max_memory_fraction=config.max_memory_fraction,
            **spec.kwargs,
        )
        normalized = _normalize_result(spec.name, raw_result)
        if normalized["error"] is not None:
            status = (
                "oom"
                if normalized["metadata"].get("failure_kind") == "oom"
                else "error"
            )
            return _failed_run(spec, repeat, normalized["error"], status=status)
        records = normalized["records"]
        timings = normalized["timings"]
        total_grid_s = float(timings.total_grid_s)
        validation = None
        if baseline_records is not None:
            validation = _validation_summary(
                compare_surfaces(
                    records,
                    baseline_records,
                    rel_tol=config.rel_tol,
                    abs_tol=config.abs_tol,
                )
            )
        return CandidateRun(
            candidate=spec.name,
            repeat=repeat,
            status="ok",
            records=records,
            timings=timings,
            total_grid_s=total_grid_s,
            per_point_latency_s=(
                total_grid_s / len(records) if records and total_grid_s is not None else None
            ),
            per_batch_eval_s=_per_batch_eval_s(timings, config.request),
            peak_cpu_memory_bytes=normalized["peak_cpu_memory_bytes"],
            peak_cuda_memory_bytes=normalized["peak_cuda_memory_bytes"],
            validation=validation,
            speedup_vs_baseline=_speedup(baseline_time_s, total_grid_s),
            metadata={
                **normalized["metadata"],
                **spec.kwargs,
                "candidate_taxonomy": _candidate_taxonomy(spec),
            },
        )
    except RuntimeError as exc:
        if _is_oom(exc):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            return _failed_run(spec, repeat, f"CUDA OOM: {exc}", status="oom")
        return _failed_run(spec, repeat, f"RuntimeError: {exc}")
    except Exception as exc:
        return _failed_run(spec, repeat, f"{type(exc).__name__}: {exc}")


def _call_runner(runner: Runner, request: SchedulerRequest, **kwargs: Any) -> Any:
    signature = inspect.signature(runner)
    accepted = {}
    accepts_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    for key, value in kwargs.items():
        if accepts_kwargs or key in signature.parameters:
            accepted[key] = value
    return runner(request, **accepted)


def _normalize_result(candidate_name: str, result: Any) -> dict[str, Any]:
    if isinstance(result, list):
        return {
            "candidate": candidate_name,
            "records": _surface(result),
            "timings": SectionTimings(),
            "peak_cpu_memory_bytes": None,
            "peak_cuda_memory_bytes": None,
            "metadata": {"normalized_from": "surface"},
            "error": None,
        }

    payload = asdict(result) if is_dataclass(result) else result
    if isinstance(payload, tuple) and payload and isinstance(payload[0], list):
        payload = {"records": payload[0]}
    if not isinstance(payload, dict):
        raise TypeError(f"unsupported candidate result type: {type(result).__name__}")

    timings = _timings(payload.get("timings"))
    return {
        "candidate": str(payload.get("candidate", candidate_name)),
        "records": _surface(payload.get("records", [])),
        "timings": timings,
        "peak_cpu_memory_bytes": _memory_bytes(
            payload,
            "peak_cpu_memory_bytes",
            "process_memory",
            "rss_bytes",
        ),
        "peak_cuda_memory_bytes": _memory_bytes(
            payload,
            "peak_cuda_memory_bytes",
            "peak_cuda_memory",
            "max_reserved_bytes",
        ),
        "metadata": dict(payload.get("metadata") or {}),
        "error": payload.get("error"),
    }


def _timings(value: Any) -> SectionTimings:
    if value is None:
        return SectionTimings()
    if isinstance(value, SectionTimings):
        return value
    payload = asdict(value) if is_dataclass(value) else value
    if isinstance(payload, dict):
        return SectionTimings(
            perturbation_s=float(payload.get("perturbation_s", 0.0)),
            binding_s=float(payload.get("binding_s", 0.0)),
            batch_eval_s=float(payload.get("batch_eval_s", 0.0)),
            total_grid_s=float(payload.get("total_grid_s", 0.0)),
        )
    raise TypeError(f"unsupported timings type: {type(value).__name__}")


def _surface(value: Any) -> Surface:
    return [
        (int(record[0]), int(record[1]), float(record[2]))
        for record in value
    ]


def _skipped_run(spec: CandidateSpec, repeat: int) -> CandidateRun:
    return CandidateRun(
        candidate=spec.name,
        repeat=repeat,
        status="skipped",
        records=[],
        timings=SectionTimings(),
        total_grid_s=None,
        per_point_latency_s=None,
        per_batch_eval_s=None,
        peak_cpu_memory_bytes=None,
        peak_cuda_memory_bytes=None,
        validation=None,
        speedup_vs_baseline=None,
        metadata={
            **spec.kwargs,
            "skip_reason": spec.skipped_reason,
            "candidate_taxonomy": _candidate_taxonomy(spec),
        },
    )


def _failed_run(
    spec: CandidateSpec,
    repeat: int,
    error: str,
    *,
    status: str = "error",
) -> CandidateRun:
    return CandidateRun(
        candidate=spec.name,
        repeat=repeat,
        status=status,
        records=[],
        timings=SectionTimings(),
        total_grid_s=None,
        per_point_latency_s=None,
        per_batch_eval_s=None,
        peak_cpu_memory_bytes=None,
        peak_cuda_memory_bytes=None,
        validation=None,
        speedup_vs_baseline=None,
        metadata={**spec.kwargs, "candidate_taxonomy": _candidate_taxonomy(spec)},
        error=error,
    )


def _candidate_taxonomy(spec: CandidateSpec) -> dict[str, Any]:
    return {
        "components": list(spec.components),
        "applies_to_sections": list(spec.applies_to_sections),
        "hypothesis": spec.hypothesis,
        "control_params": dict(spec.kwargs),
    }


_SURFACE_MAX_ABS_BUDGET = 1e-4  # float32 rounding budget per spec


def _validation_summary(comparison: Any) -> dict[str, Any]:
    return {
        "point_count": comparison.point_count,
        "mismatch_count": comparison.mismatch_count,
        "max_abs_error": comparison.max_abs_error,
        "max_abs_within_budget": comparison.max_abs_error <= _SURFACE_MAX_ABS_BUDGET,
        "surface_budget": _SURFACE_MAX_ABS_BUDGET,
        "rmse": comparison.rmse,
        "allclose": comparison.allclose,
        "rel_tol": comparison.rel_tol,
        "abs_tol": comparison.abs_tol,
        "first_mismatches": [asdict(item) for item in comparison.first_mismatches],
    }


def _summarize_runs(runs: list[CandidateRun]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[CandidateRun]] = {}
    for run in runs:
        by_candidate.setdefault(run.candidate, []).append(run)

    summaries = []
    for candidate, candidate_runs in by_candidate.items():
        ok_times = [
            run.total_grid_s
            for run in candidate_runs
            if run.status == "ok" and run.total_grid_s is not None
        ]
        speedups = [
            run.speedup_vs_baseline
            for run in candidate_runs
            if run.speedup_vs_baseline is not None
        ]
        validations = [
            run.validation["allclose"]
            for run in candidate_runs
            if run.validation is not None
        ]
        summaries.append(
            {
                "candidate": candidate,
                "taxonomy": _summary_taxonomy(candidate_runs),
                "status_counts": _status_counts(candidate_runs),
                "mean_total_grid_s": _mean(ok_times),
                "stdev_total_grid_s": _stdev(ok_times),
                "mean_speedup_vs_baseline": _mean(speedups),
                "all_validations_passed": all(validations) if validations else None,
            }
        )
    return summaries


def _summary_taxonomy(runs: list[CandidateRun]) -> dict[str, Any] | None:
    for run in runs:
        taxonomy = run.metadata.get("candidate_taxonomy")
        if isinstance(taxonomy, dict):
            return taxonomy
    return None


def _status_counts(runs: list[CandidateRun]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for run in runs:
        counts[run.status] = counts.get(run.status, 0) + 1
    return counts


def _write_summary(summary: dict[str, Any], run_label: str | None = None) -> Path:
    output_dir = Path("outputs") / "functional_eval"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"{_filename_label(run_label)}{timestamp}-summary.json"
    summary["output_path"] = str(output_path)
    output_path.write_text(
        json.dumps(_json_safe(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return output_path


def _print_table(summary: dict[str, Any]) -> None:
    platform = summary["platform"]
    print(
        "functional_eval experiment "
        f"backend={platform['device_type']} "
        f"device={platform.get('device_name') or 'n/a'}"
    )
    print(f"summary={summary['output_path']}")
    print("candidate                 status       mean_s   speedup   valid")
    for row in summary["candidate_summary"]:
        status = ",".join(
            f"{key}:{value}" for key, value in sorted(row["status_counts"].items())
        )
        mean_s = _format_float(row["mean_total_grid_s"])
        speedup = _format_float(row["mean_speedup_vs_baseline"])
        valid = row["all_validations_passed"]
        print(
            f"{row['candidate']:<25} "
            f"{status:<12.12} "
            f"{mean_s:>8} "
            f"{speedup:>9} "
            f"{str(valid):>7}"
        )


def _request_summary(config: FunctionalEvalConfig) -> dict[str, Any]:
    request = config.request
    return {
        "seed": config.seed,
        "point_chunk_sizes": list(config.point_chunk_sizes),
        "rel_tol": config.rel_tol,
        "abs_tol": config.abs_tol,
        "max_memory_fraction": config.max_memory_fraction,
        "run_label": config.run_label,
        "workload_name": request.task.name,
        "task_name": request.task.name,
        "task": request.task.task,
        "model": request.task.model,
        "model_family": request.task.model,
        "loss": request.task.loss,
        "dataset": {
            "name": request.task.dataset.name,
            "path": request.task.dataset.path,
            "input_shape": list(request.task.dataset.input_shape),
            "sample_count": request.task.dataset.sample_count,
        },
        "dataset_name": request.task.dataset.name,
        "dataset_sample_count": request.task.dataset.sample_count,
        "checkpoint_path": request.task.checkpoint_path,
        "grid_resolution": request.grid.resolution,
        "grid_scale": request.grid.scale,
        "mode": request.mode._tag,
        "gpu_batch_size": request.mode.gpu_batch_size,
        "device": request.device,
    }


def _platform_metadata(device: torch.device) -> dict[str, Any]:
    device_name = None
    cuda = None
    if device.type == "cuda" and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(device)
        props = torch.cuda.get_device_properties(device)
        cuda = {
            "device_count": int(torch.cuda.device_count()),
            "current_device": int(torch.cuda.current_device()),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": int(props.total_memory),
            "multi_processor_count": int(props.multi_processor_count),
        }
    return {
        "device_type": device.type,
        "device_name": device_name,
        "cuda": cuda,
        "torch_version": torch.__version__,
    }


def _resolve_device(device: str) -> torch.device:
    return torch.device(
        device
        if device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )


def _per_batch_eval_s(timings: SectionTimings, request: SchedulerRequest) -> float | None:
    batch_size = max(1, int(request.mode.gpu_batch_size))
    batches = math.ceil(request.task.dataset.sample_count / batch_size)
    point_count = request.grid.resolution * request.grid.resolution
    denominator = max(1, batches * point_count)
    return timings.batch_eval_s / denominator if timings.batch_eval_s > 0 else None


def _speedup(baseline_s: float | None, candidate_s: float | None) -> float | None:
    if baseline_s is None or candidate_s is None or candidate_s <= 0:
        return None
    return baseline_s / candidate_s


def _mean(values: Any) -> float | None:
    concrete = [float(value) for value in values if value is not None]
    return statistics.fmean(concrete) if concrete else None


def _stdev(values: Any) -> float | None:
    concrete = [float(value) for value in values if value is not None]
    return statistics.stdev(concrete) if len(concrete) > 1 else None


def _memory_bytes(
    payload: dict[str, Any],
    direct_key: str,
    snapshot_key: str,
    snapshot_field: str,
) -> int | None:
    direct_value = payload.get(direct_key)
    if direct_value is not None:
        return int(direct_value)
    snapshot = payload.get(snapshot_key)
    if isinstance(snapshot, dict):
        value = snapshot.get(snapshot_field)
        return None if value is None else int(value)
    return None


def _is_oom(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "out of memory" in message or "cuda oom" in message


def _format_float(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _filename_label(value: str | None) -> str:
    if value is None or not value.strip():
        return ""
    safe = "".join(
        char if char.isalnum() or char in ("-", "_") else "-"
        for char in value.strip().lower()
    ).strip("-_")
    return f"{safe}-" if safe else ""


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def build_config(**overrides: Any) -> SimpleNamespace:
    defaults = {
        "device": "auto",
        "seed": 1337,
        "resolution": 8,
        "scale": 1.0,
        "batch_size": 32,
        "sample_count": None,
        "workload_name": DEFAULT_WORKLOAD_NAME,
        "run_label": None,
        "point_chunk_sizes": [2, 4, 8, 16, 32, 64],
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def main() -> None:
    args = build_config()
    request = build_default_request(
        workload_name=args.workload_name,
        device=args.device,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        resolution=args.resolution,
        scale=args.scale,
    )
    config = FunctionalEvalConfig(
        request=request,
        seed=args.seed,
        point_chunk_sizes=tuple(args.point_chunk_sizes),
        run_label=args.run_label,
    )
    run_experiment(config)


if __name__ == "__main__":
    main()
