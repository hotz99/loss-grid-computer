#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.backends import run_backend
from src.backends.base import resolve_device
from src.calibration import (
    resolve_available_cpu_cores,
    resolve_cpu_batch_size_candidates,
    resolve_cpu_worker_candidates,
    run_calibration,
)
from src.compare import compare_surfaces
from src.system_schema import GridSpec, HybridMode, MLTaskSpec, SchedulerRequest, VanillaMode
from src.workloads import WORKLOADS


ROW_GRU_WORKLOAD = "cifar10_row_gru_classification"
RESNET_WORKLOAD = "cifar10_resnet20_classification"


def collect_rq3_metrics(args: argparse.Namespace) -> dict[str, Any]:
    requested_device = args.device
    resolved_device = resolve_device(requested_device)
    platform = _platform_summary(requested_device, resolved_device)
    grid = GridSpec(args.grid_resolution, args.grid_scale)
    output_path = _output_path(args.output)

    if resolved_device.type == "cpu" and not args.allow_cpu_only:
        summary = {
            "schema_version": "rq3-workload-affinity-v1",
            "created_at": _now(),
            "status": "skipped",
            "skip_reason": (
                "hybrid execution requires an accelerator device; rerun on CUDA or MPS, "
                "or pass --allow-cpu-only to collect metadata only"
            ),
            "platform": platform,
            "config": _config_summary(args),
            "workloads": {},
        }
        _write_summary(output_path, summary)
        summary["output_path"] = str(output_path)
        return summary

    if resolved_device.type == "cpu":
        workload_names = [ROW_GRU_WORKLOAD]
        if args.include_resnet:
            workload_names.insert(0, RESNET_WORKLOAD)
        workloads = {
            name: _collect_cpu_only_metadata(
                _task_spec(name, args.sample_count),
                grid,
                args,
            )
            for name in workload_names
        }
        summary = {
            "schema_version": "rq3-workload-affinity-v1",
            "created_at": _now(),
            "status": "metadata_only",
            "skip_reason": "hybrid execution requires an accelerator device",
            "platform": platform,
            "config": _config_summary(args),
            "workloads": workloads,
        }
        _write_summary(output_path, summary)
        summary["output_path"] = str(output_path)
        return summary

    workload_names = [ROW_GRU_WORKLOAD]
    if args.include_resnet:
        workload_names.insert(0, RESNET_WORKLOAD)

    workloads = {
        name: _collect_workload_metrics(
            task_spec=_task_spec(name, args.sample_count),
            grid=grid,
            args=args,
            resolved_device=resolved_device,
        )
        for name in workload_names
    }

    summary = {
        "schema_version": "rq3-workload-affinity-v1",
        "created_at": _now(),
        "status": "completed",
        "platform": platform,
        "config": _config_summary(args),
        "workloads": workloads,
        "rq3_minimal_metrics": _rq3_minimal_metrics(workloads),
    }
    _write_summary(output_path, summary)
    summary["output_path"] = str(output_path)
    return summary


def _collect_workload_metrics(
    *,
    task_spec: MLTaskSpec,
    grid: GridSpec,
    args: argparse.Namespace,
    resolved_device: torch.device,
) -> dict[str, Any]:
    del resolved_device
    vanilla_request = SchedulerRequest(
        task_spec,
        grid,
        VanillaMode(args.gpu_batch_size),
        args.device,
    )
    sweep_mode = HybridMode(
        args.gpu_batch_size,
        args.cpu_batch_size,
        args.cpu_workers,
    )
    sweep_request = SchedulerRequest(task_spec, grid, sweep_mode, args.device)

    vanilla_base = run_backend(vanilla_request, args.seed)
    sweep = _sample_slowdowns(
        sweep_request=sweep_request,
        vanilla_surface=vanilla_base.records,
        vanilla_base_total_s=vanilla_base.record.measurement.total_s,
        args=args,
    )
    fixed_slowdown = _select_fixed_slowdown(sweep, args.fixed_slowdown)

    fixed_vanilla = run_backend(vanilla_request, args.seed, fixed_slowdown)
    cpu_worker_candidates = _bounded_cpu_worker_candidates(args.max_cpu_worker_candidate)
    cpu_batch_candidates = resolve_cpu_batch_size_candidates(
        task_spec.dataset.sample_count,
        args.gpu_batch_size,
    )
    selected_mode = run_calibration(
        SchedulerRequest(task_spec, grid, HybridMode(args.gpu_batch_size), args.device),
        fixed_vanilla.record.measurement.total_s,
        cpu_worker_candidates,
        cpu_batch_candidates,
        args.calibration_retry,
        args.seed,
        fixed_slowdown,
    )

    evaluated_hybrid_mode = (
        selected_mode
        if isinstance(selected_mode, HybridMode)
        else HybridMode(args.gpu_batch_size, args.cpu_batch_size, args.cpu_workers)
    )
    fixed_hybrid = run_backend(
        SchedulerRequest(task_spec, grid, evaluated_hybrid_mode, args.device),
        args.seed,
        fixed_slowdown,
    )
    surface = compare_surfaces(
        fixed_vanilla.records,
        fixed_hybrid.records,
        args.atol,
        args.rtol,
        fixed_vanilla.record.measurement.total_s,
        fixed_hybrid.record.measurement.total_s,
    )

    return {
        "workload_name": task_spec.name,
        "model_family": task_spec.model,
        "task": task_spec.task,
        "loss": task_spec.loss,
        "dataset": asdict(task_spec.dataset),
        "checkpoint_path": task_spec.checkpoint_path,
        "slowdown_sweep": sweep,
        "fixed_regime": {
            "slowdown": fixed_slowdown,
            "calibration_policy": asdict(selected_mode),
            "evaluated_hybrid_policy": asdict(evaluated_hybrid_mode),
            "vanilla_runtime_s": fixed_vanilla.record.measurement.total_s,
            "hybrid_runtime_s": fixed_hybrid.record.measurement.total_s,
            "speedup_vs_vanilla": surface["speedup_rhs_vs_lhs_baseline"],
            "worker_throughput_split": _worker_split(fixed_hybrid.runtime_log),
            "surface_validation": {
                "allclose": surface["allclose"],
                "rmse": surface["rmse"],
                "mismatch_count": surface["mismatch_count"],
                "atol": surface["atol"],
                "rtol": surface["rtol"],
            },
        },
    }


def _collect_cpu_only_metadata(
    task_spec: MLTaskSpec,
    grid: GridSpec,
    args: argparse.Namespace,
) -> dict[str, Any]:
    request = SchedulerRequest(
        task_spec,
        grid,
        VanillaMode(args.gpu_batch_size),
        args.device,
    )
    result = run_backend(request, args.seed)
    return {
        "workload_name": task_spec.name,
        "model_family": task_spec.model,
        "task": task_spec.task,
        "loss": task_spec.loss,
        "dataset": asdict(task_spec.dataset),
        "checkpoint_path": task_spec.checkpoint_path,
        "vanilla_runtime_s": result.record.measurement.total_s,
        "note": "CPU-only run records metadata and vanilla runtime; RQ3 hybrid metrics require CUDA or MPS.",
    }


def _sample_slowdowns(
    *,
    sweep_request: SchedulerRequest,
    vanilla_surface: list[tuple[int, int, float]],
    vanilla_base_total_s: float,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sampled = []
    slowdown = args.min_slowdown
    previous = None
    crossover = None

    while slowdown <= args.max_slowdown + 1e-9:
        result = run_backend(sweep_request, args.seed, slowdown)
        scaled_vanilla_total_s = vanilla_base_total_s * slowdown
        surface = compare_surfaces(
            vanilla_surface,
            result.records,
            args.atol,
            args.rtol,
            scaled_vanilla_total_s,
            result.record.measurement.total_s,
        )
        speedup = surface["speedup_rhs_vs_lhs_baseline"]
        point = {
            "slowdown": slowdown,
            "hybrid_runtime_s": result.record.measurement.total_s,
            "scaled_vanilla_runtime_s": scaled_vanilla_total_s,
            "speedup_vs_scaled_vanilla": speedup,
            "hybrid_wins": bool(speedup is not None and speedup > 1.0),
            "surface_valid": surface["allclose"],
            "worker_throughput_split": _worker_split(result.runtime_log),
        }
        sampled.append(point)

        if (
            previous is not None
            and not previous["hybrid_wins"]
            and point["hybrid_wins"]
            and crossover is None
        ):
            crossover = (previous["slowdown"], point["slowdown"])

        previous = point
        slowdown *= args.slowdown_step
        if args.slowdown_step <= 1.0:
            break

    return {
        "crossover_region": list(crossover) if crossover else None,
        "sampled_slowdown_points": sampled,
        "best_sampled_slowdown": max(
            sampled,
            key=lambda point: point["speedup_vs_scaled_vanilla"] or float("-inf"),
        )["slowdown"],
    }


def _select_fixed_slowdown(sweep: dict[str, Any], explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    crossover = sweep["crossover_region"]
    if crossover:
        return statistics.fmean(crossover)
    return float(sweep["best_sampled_slowdown"])


def _worker_split(runtime_log: dict[str, Any]) -> dict[str, Any]:
    execution = runtime_log.get("hybrid_execution", {})
    gpu_points = int(execution.get("gpu_points_processed", 0))
    cpu_points = int(execution.get("cpu_points_processed", 0))
    total_points = gpu_points + cpu_points
    return {
        "gpu_points_processed": gpu_points,
        "cpu_points_processed": cpu_points,
        "gpu_point_fraction": _fraction(gpu_points, total_points),
        "cpu_point_fraction": _fraction(cpu_points, total_points),
        "gpu_throughput_points_per_s": execution.get("gpu_throughput_points_per_s"),
        "cpu_throughput_points_per_s": execution.get("cpu_throughput_points_per_s"),
        "total_throughput_points_per_s": execution.get("throughput_points_per_s"),
    }


def _rq3_minimal_metrics(workloads: dict[str, Any]) -> dict[str, Any]:
    row_gru = workloads.get(ROW_GRU_WORKLOAD)
    if row_gru is None:
        return {}

    fixed = row_gru["fixed_regime"]
    metrics = {
        "workload_name": row_gru["workload_name"],
        "model_family": row_gru["model_family"],
        "loss_function": row_gru["loss"],
        "crossover_region": row_gru["slowdown_sweep"]["crossover_region"],
        "sampled_slowdown_points": [
            point["slowdown"]
            for point in row_gru["slowdown_sweep"]["sampled_slowdown_points"]
        ],
        "selected_calibration_setting": fixed["calibration_policy"],
        "vanilla_runtime_s": fixed["vanilla_runtime_s"],
        "hybrid_runtime_s": fixed["hybrid_runtime_s"],
        "worker_level_throughput_split": fixed["worker_throughput_split"],
        "surface_validation": fixed["surface_validation"],
    }

    if RESNET_WORKLOAD in workloads:
        metrics["comparison_workload"] = {
            "workload_name": RESNET_WORKLOAD,
            "model_family": workloads[RESNET_WORKLOAD]["model_family"],
            "fixed_regime": workloads[RESNET_WORKLOAD]["fixed_regime"],
            "crossover_region": workloads[RESNET_WORKLOAD]["slowdown_sweep"][
                "crossover_region"
            ],
        }
    return metrics


def _task_spec(name: str, sample_count: int) -> MLTaskSpec:
    spec = WORKLOADS[name].spec
    checkpoint = spec.checkpoint_path
    if name == RESNET_WORKLOAD:
        checkpoint = "assets/cifar10-resnet20-0.pkl"
    if name == ROW_GRU_WORKLOAD:
        checkpoint = "assets/cifar10-row-gru-0.pkl"
    return replace(
        spec,
        dataset=replace(spec.dataset, sample_count=sample_count),
        checkpoint_path=checkpoint,
    )


def _bounded_cpu_worker_candidates(max_candidate: int | None) -> tuple[int, ...]:
    values = resolve_cpu_worker_candidates()
    if max_candidate is None:
        return values
    bounded = tuple(value for value in values if value <= max_candidate)
    return bounded or (min(resolve_available_cpu_cores(), max_candidate),)


def _platform_summary(requested_device: str, resolved_device: torch.device) -> dict[str, Any]:
    return {
        "requested_device": requested_device,
        "resolved_device": str(resolved_device),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "cpu_cores": resolve_available_cpu_cores(),
        "torch_version": torch.__version__,
    }


def _config_summary(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "sample_count": args.sample_count,
        "grid_resolution": args.grid_resolution,
        "grid_scale": args.grid_scale,
        "gpu_batch_size": args.gpu_batch_size,
        "cpu_batch_size": args.cpu_batch_size,
        "cpu_workers": args.cpu_workers,
        "min_slowdown": args.min_slowdown,
        "max_slowdown": args.max_slowdown,
        "slowdown_step": args.slowdown_step,
        "fixed_slowdown": args.fixed_slowdown,
        "include_resnet": args.include_resnet,
        "calibration_retry": args.calibration_retry,
        "max_cpu_worker_candidate": args.max_cpu_worker_candidate,
        "atol": args.atol,
        "rtol": args.rtol,
    }


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _output_path(raw: str | None) -> Path:
    if raw:
        return Path(raw)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("outputs") / "rq3" / f"{timestamp}-summary.json"


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect the minimal RQ3 workload-affinity metrics for CIFAR-10 row-GRU, "
            "optionally with a ResNet20 comparison workload."
        )
    )
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--allow-cpu-only", action="store_true")
    parser.add_argument("--include-resnet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sample-count", type=int, default=1024)
    parser.add_argument("--grid-resolution", type=int, default=8)
    parser.add_argument("--grid-scale", type=float, default=1.0)
    parser.add_argument("--gpu-batch-size", type=int, default=64)
    parser.add_argument("--cpu-batch-size", type=int, default=4)
    parser.add_argument("--cpu-workers", type=int, default=2)
    parser.add_argument("--min-slowdown", type=float, default=1.0)
    parser.add_argument("--max-slowdown", type=float, default=20.0)
    parser.add_argument("--slowdown-step", type=float, default=1.8)
    parser.add_argument("--fixed-slowdown", type=float)
    parser.add_argument("--calibration-retry", type=int, default=1)
    parser.add_argument("--max-cpu-worker-candidate", type=int)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = build_parser().parse_args(argv)
    summary = collect_rq3_metrics(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
