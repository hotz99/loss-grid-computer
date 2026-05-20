from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.dont_write_bytecode = True
os.environ.setdefault("LGC_VERBOSE_EXPERIMENT_LOGS", "0")

from src.backends.vanilla import run as run_vanilla_backend
from src.functional_eval.compiled import (
    run_compiled_forward_surface,
    run_compiled_functional_surface,
    run_compiled_vmapped_surface,
)
from src.functional_eval.experiment import build_default_request
from src.functional_eval.sequential import run_functional_sequential_surface
from src.functional_eval.validation import compare_surfaces
from src.functional_eval.vmapped import run_vmapped_surface


DEFAULT_WORKLOADS = (
    "cifar10_resnet20_classification",
    "cifar10_row_gru_classification",
    "california_mlp_regression",
    "mnist_mlp_classification",
)

DEFAULT_CANDIDATES = (
    "compiled_forward",
    "compiled_functional",
    "compiled_vmapped_chunk_32",
    "compiled_vmapped_chunk_64",
)


def main() -> int:
    args = _parse_args()
    _emit(
        "mvp_start",
        grid_resolution=args.grid_resolution,
        grid_scale=args.grid_scale,
        variant_count=args.variants,
        device=args.device,
        workloads=args.workloads,
        candidates=args.candidates,
        compile_mode=args.compile_mode,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        calibration_s=args.calibration_s,
        note="stdout-only records; no experiment result files are written",
    )

    for workload in args.workloads:
        request = build_default_request(
            workload_name=workload,
            device=args.device,
            sample_count=args.sample_count,
            batch_size=args.batch_size,
            resolution=args.grid_resolution,
            scale=args.grid_scale,
        )
        _run_workload(request, args)

    _emit("mvp_complete")
    return 0


def _run_workload(request, args: argparse.Namespace) -> None:
    _emit(
        "workload_start",
        workload=request.task.name,
        grid_resolution=request.grid.resolution,
        sample_count=request.task.dataset.sample_count,
        batch_size=request.mode.gpu_batch_size,
    )

    baseline_started = perf_counter()
    baseline = run_vanilla_backend(request, seed=args.seed, profile_sections=True)
    baseline_s = float(baseline.record.measurement.total_s)
    baseline_elapsed_s = perf_counter() - baseline_started
    baseline_records = baseline.records
    vanilla_session_s = args.variants * baseline_s
    valid_options: list[dict[str, Any]] = [
        {
            "candidate": "vanilla_eager",
            "grid_s": baseline_s,
            "session_s": vanilla_session_s,
            "validation_pass": True,
        }
    ]
    _emit(
        "measurement",
        workload=request.task.name,
        candidate="vanilla_eager",
        status="ok",
        grid_s=baseline_s,
        elapsed_s=baseline_elapsed_s,
        session_s=vanilla_session_s,
        variant_count=args.variants,
        validation_pass=True,
        section_timings=_json_safe(baseline.runtime_log.get("section_timings")),
    )

    eager_results: dict[str, Any] = {}
    for compiled_candidate in args.candidates:
        eager_spec = _eager_comparison_spec(compiled_candidate)
        if eager_spec is None or eager_spec[0] in eager_results:
            continue
        eager_name, eager_runner, eager_kwargs = eager_spec
        eager_results[eager_name] = _run_eager_comparison(
            request,
            args,
            eager_name,
            eager_runner,
            eager_kwargs,
            baseline_records,
            baseline_s,
        )
        eager_summary = eager_results[eager_name]
        if eager_summary is not None and eager_summary["validation_pass"]:
            valid_options.append(
                {
                    "candidate": eager_name,
                    "grid_s": eager_summary["grid_s"],
                    "session_s": eager_summary["session_s"],
                    "validation_pass": True,
                }
            )

    for candidate in args.candidates:
        compiled_summary = _run_compiled_candidate(
            request,
            args,
            candidate,
            baseline_records,
            baseline_s,
            vanilla_session_s,
            eager_results,
        )
        if compiled_summary is not None and compiled_summary["validation_pass"]:
            valid_options.append(
                {
                    "candidate": candidate,
                    "grid_s": compiled_summary["steady_grid_s"],
                    "session_s": compiled_summary["session_s"],
                    "validation_pass": True,
                }
            )

    best = min(valid_options, key=lambda item: item["session_s"])
    _emit(
        "workload_summary",
        workload=request.task.name,
        best_candidate=best["candidate"],
        best_session_s=best["session_s"],
        best_grid_s=best["grid_s"],
        valid_candidate_count=len(valid_options),
        speedup_vs_vanilla_session=vanilla_session_s / best["session_s"]
        if best["session_s"] > 0
        else None,
        valid_options=valid_options,
    )
    _emit("workload_complete", workload=request.task.name)


def _run_eager_comparison(
    request,
    args: argparse.Namespace,
    candidate: str,
    runner: Callable[..., Any],
    kwargs: dict[str, Any],
    baseline_records,
    baseline_s: float,
) -> dict[str, Any] | None:
    _emit("run_start", workload=request.task.name, candidate=candidate)
    started = perf_counter()
    try:
        result = runner(request, seed=args.seed, **kwargs)
        elapsed_s = perf_counter() - started
        validation = compare_surfaces(
            baseline_records,
            result.records,
            rel_tol=args.rel_tol,
            abs_tol=args.abs_tol,
        )
        grid_s = float(result.timings.total_grid_s)
        _emit(
            "measurement",
            workload=request.task.name,
            candidate=candidate,
            status="ok",
            grid_s=grid_s,
            elapsed_s=elapsed_s,
            session_s=args.variants * grid_s,
            speedup_vs_vanilla=baseline_s / grid_s if grid_s > 0 else None,
            validation_pass=validation.allclose,
            validation=_json_safe(validation),
            section_timings=_json_safe(result.timings),
        )
        return {
            "result": result,
            "grid_s": grid_s,
            "session_s": args.variants * grid_s,
            "validation_pass": validation.allclose,
        }
    except Exception as exc:
        _emit(
            "measurement",
            workload=request.task.name,
            candidate=candidate,
            status="error",
            elapsed_s=perf_counter() - started,
            error=f"{type(exc).__name__}: {exc}",
        )
        return None


def _run_compiled_candidate(
    request,
    args: argparse.Namespace,
    candidate: str,
    baseline_records,
    baseline_s: float,
    vanilla_session_s: float,
    eager_results: dict[str, Any],
) -> dict[str, Any] | None:
    runner, kwargs = _compiled_runner(candidate)
    kwargs = {**kwargs, "compile_mode": _compile_mode_arg(args)}
    _emit("run_start", workload=request.task.name, candidate=candidate)
    started = perf_counter()
    result = runner(request, seed=args.seed, **kwargs)
    elapsed_s = perf_counter() - started

    if result.error is not None:
        _emit(
            "measurement",
            workload=request.task.name,
            candidate=candidate,
            status="error",
            elapsed_s=elapsed_s,
            compile_s=result.compile_s,
            graph_break_count=result.graph_break_count,
            recompile_count=result.recompile_count,
            compile_counters=result.compile_counters,
            metadata=result.metadata,
            error=result.error,
        )
        return None

    validation = compare_surfaces(
        baseline_records,
        result.records,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
    )
    compiled_session_s = (
        result.compile_s
        + args.calibration_s
        + args.variants * result.steady_grid_s
    )
    eager_candidate_name = _corresponding_eager_candidate(candidate)
    eager_result = eager_results.get(eager_candidate_name) if eager_candidate_name else None
    eager_grid_s = None if eager_result is None else float(eager_result["grid_s"])
    eager_session_s = None if eager_grid_s is None else args.variants * eager_grid_s
    runtime_claim_eligible = validation.allclose
    beats_vanilla_runtime_only = compiled_session_s < vanilla_session_s
    beats_eager_runtime_only = (
        None if eager_session_s is None else compiled_session_s < eager_session_s
    )

    _emit(
        "measurement",
        workload=request.task.name,
        candidate=candidate,
        status="ok",
        compile_s=result.compile_s,
        first_call_s=result.first_call_s,
        steady_grid_s=result.steady_grid_s,
        elapsed_s=elapsed_s,
        session_s=compiled_session_s,
        calibration_s=args.calibration_s,
        vanilla_session_s=vanilla_session_s,
        eager_candidate=eager_candidate_name,
        eager_session_s=eager_session_s,
        variant_count=args.variants,
        speedup_vs_vanilla_grid=baseline_s / result.steady_grid_s
        if result.steady_grid_s > 0
        else None,
        speedup_vs_vanilla_session=vanilla_session_s / compiled_session_s
        if compiled_session_s > 0
        else None,
        speedup_vs_eager_session=eager_session_s / compiled_session_s
        if eager_session_s is not None and compiled_session_s > 0
        else None,
        beats_vanilla=runtime_claim_eligible and beats_vanilla_runtime_only,
        beats_vanilla_runtime_only=beats_vanilla_runtime_only,
        beats_corresponding_eager=(
            None
            if eager_session_s is None
            else runtime_claim_eligible and beats_eager_runtime_only
        ),
        beats_corresponding_eager_runtime_only=beats_eager_runtime_only,
        runtime_claim_eligible=runtime_claim_eligible,
        validation_pass=validation.allclose,
        validation=_json_safe(validation),
        graph_break_count=result.graph_break_count,
        recompile_count=result.recompile_count,
        compile_counters=result.compile_counters,
        section_timings=_json_safe(result.timings),
        metadata=result.metadata,
    )
    return {
        "candidate": candidate,
        "steady_grid_s": result.steady_grid_s,
        "session_s": compiled_session_s,
        "validation_pass": validation.allclose,
        "runtime_claim_eligible": runtime_claim_eligible,
    }


def _compiled_runner(candidate: str) -> tuple[Callable[..., Any], dict[str, Any]]:
    if candidate == "compiled_forward":
        return run_compiled_forward_surface, {}
    if candidate == "compiled_functional":
        return run_compiled_functional_surface, {}
    chunk_size = _parse_compiled_vmapped_chunk(candidate)
    if chunk_size is not None:
        return run_compiled_vmapped_surface, {"point_chunk_size": chunk_size}
    raise ValueError(f"unknown compiled candidate: {candidate}")


def _corresponding_eager_candidate(candidate: str) -> str | None:
    if candidate == "compiled_functional":
        return "functional_sequential"
    chunk_size = _parse_compiled_vmapped_chunk(candidate)
    if chunk_size is not None:
        return f"vmapped_chunk_{chunk_size}"
    return None


def _eager_comparison_spec(
    candidate: str,
) -> tuple[str, Callable[..., Any], dict[str, Any]] | None:
    if candidate == "compiled_functional":
        return (
            "functional_sequential",
            run_functional_sequential_surface,
            {},
        )
    chunk_size = _parse_compiled_vmapped_chunk(candidate)
    if chunk_size is not None:
        return (
            f"vmapped_chunk_{chunk_size}",
            run_vmapped_surface,
            {"point_chunk_size": chunk_size},
        )
    return None


def _parse_compiled_vmapped_chunk(candidate: str) -> int | None:
    prefix = "compiled_vmapped_chunk_"
    if not candidate.startswith(prefix):
        return None
    chunk_text = candidate.removeprefix(prefix)
    try:
        chunk_size = int(chunk_text)
    except ValueError:
        raise ValueError(f"invalid vmapped chunk candidate: {candidate}") from None
    if chunk_size < 1:
        raise ValueError(f"vmapped chunk size must be >= 1: {candidate}")
    return chunk_size


def _compile_mode_arg(args: argparse.Namespace) -> str | None:
    return None if args.compile_mode == "default" else args.compile_mode


def _emit(event: str, **payload: Any) -> None:
    print(json.dumps({"event": event, **_json_safe(payload)}, sort_keys=True), flush=True)


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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stdout-only torch.compile MVP for loss-grid evaluation.",
    )
    parser.add_argument(
        "--workload",
        dest="workloads",
        action="append",
        choices=DEFAULT_WORKLOADS,
        help="Workload to run. Repeat to select multiple. Defaults to all four.",
    )
    parser.add_argument(
        "--candidate",
        dest="candidates",
        action="append",
        help="Compiled candidate to run. Repeat to select multiple. Defaults to all.",
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune-no-cudagraphs"),
        default="default",
        help="torch.compile mode for compiled candidates.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
    )
    parser.add_argument("--grid-resolution", type=int, default=8)
    parser.add_argument("--grid-scale", type=float, default=1.0)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--variants", type=int, default=4)
    parser.add_argument(
        "--calibration-s",
        type=float,
        default=0.0,
        help=(
            "Calibration overhead to include in compiled session accounting. "
            "The MVP does not run calibration itself."
        ),
    )
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--rel-tol", type=float, default=1e-5)
    parser.add_argument("--abs-tol", type=float, default=1e-6)
    args = parser.parse_args()
    args.workloads = tuple(args.workloads or DEFAULT_WORKLOADS)
    args.candidates = tuple(args.candidates or DEFAULT_CANDIDATES)
    for candidate in args.candidates:
        try:
            _compiled_runner(candidate)
        except ValueError as exc:
            parser.error(str(exc))
    if args.grid_resolution != 8:
        _emit(
            "config_warning",
            message="MVP protocol requested grid=8x8; override accepted for smoke testing only",
            grid_resolution=args.grid_resolution,
        )
    if args.variants < 1:
        raise SystemExit("--variants must be >= 1")
    if args.calibration_s < 0:
        raise SystemExit("--calibration-s must be >= 0")
    return args


if __name__ == "__main__":
    raise SystemExit(main())
