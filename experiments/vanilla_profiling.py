from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.common import (
    configured_workloads,
    get_shared_artifact,
    shared_artifact_key,
    unavailable_payload,
    workload_metadata,
    workload_unavailable_reason,
)
from src.backends.vanilla import run as run_vanilla_backend
from src.functional_eval.experiment import build_default_request
from src.functional_eval.memory import SectionTimings


def compute_vanilla_full_grid(
    workload_name: str,
    config: SimpleNamespace,
) -> tuple[str, Any, dict[str, Any]] | None:
    if workload_unavailable_reason(workload_name) is not None:
        return None
    request = build_default_request(
        workload_name=workload_name,
        device=config.device,
        sample_count=config.sample_count,
        batch_size=config.functional_eval_batch_size,
        resolution=config.grid_resolution,
        scale=config.grid_scale,
    )
    result = run_vanilla_backend(request, seed=config.seed, profile_sections=True)
    key = shared_artifact_key(
        node_type="vanilla_full_grid",
        workload_name=workload_name,
        checkpoint_path=request.task.checkpoint_path,
        sample_count=request.task.dataset.sample_count,
        grid_resolution=request.grid.resolution,
        grid_scale=request.grid.scale,
        device=request.device,
        seed=config.seed,
        gpu_batch_size=request.mode.gpu_batch_size,
        slowdown_factor=1.0,
    )
    summary = {
        "node_type": "vanilla_full_grid",
        "workload": workload_name,
        "total_grid_s": result.record.measurement.total_s,
        "point_count": len(result.records),
    }
    return key, result, summary


def collect(config: SimpleNamespace, shared_state: dict[str, Any] | None = None) -> dict[str, Any]:
    if shared_state is None:
        shared_state = {}
    workloads: dict[str, Any] = {}
    for workload_name in configured_workloads(config):
        unavailable_reason = workload_unavailable_reason(workload_name)
        if unavailable_reason is not None:
            workloads[workload_name] = unavailable_payload(
                workload_name,
                unavailable_reason,
                config.sample_count,
            )
            continue

        try:
            request = build_default_request(
                workload_name=workload_name,
                device=config.device,
                sample_count=config.sample_count,
                batch_size=config.functional_eval_batch_size,
                resolution=config.grid_resolution,
                scale=config.grid_scale,
            )
            key = shared_artifact_key(
                node_type="vanilla_full_grid",
                workload_name=workload_name,
                checkpoint_path=request.task.checkpoint_path,
                sample_count=request.task.dataset.sample_count,
                grid_resolution=request.grid.resolution,
                grid_scale=request.grid.scale,
                device=request.device,
                seed=config.seed,
                gpu_batch_size=request.mode.gpu_batch_size,
                slowdown_factor=1.0,
            )
            result = get_shared_artifact(shared_state, key)
            if result is None:
                result = run_vanilla_backend(request, seed=config.seed, profile_sections=True)
            total = result.record.measurement.total_s
            section_timings = dict(result.runtime_log.get("section_timings") or {})
            section_timings["total_grid_s"] = total
            workloads[workload_name] = {
                "status": "completed",
                "workload": workload_metadata(workload_name, config.sample_count),
                "candidate": "baseline_original",
                "section_timings": asdict(SectionTimings(**section_timings)),
                "total_grid_s": total,
                "per_point_latency_s": (
                    None
                    if total is None
                    else total / max(1, request.grid.resolution**2)
                ),
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
        except Exception as exc:
            workloads[workload_name] = {
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
                "workload": workload_metadata(workload_name, config.sample_count),
            }

    completed = {
        name: item.get("total_grid_s")
        for name, item in workloads.items()
        if item.get("status") == "completed"
    }
    return {
        "schema_version": "experiment-a-vanilla-profiling-v1",
        "workload_count": len(workloads),
        "workloads": workloads,
        "record": {
            "status": "completed",
            "workload_count": len(workloads),
            "completed_workloads": len(completed),
            "total_grid_s_by_workload": completed,
        },
    }


def run(
    config: SimpleNamespace,
    output_dir: Path,
    shared_state: dict[str, Any],
) -> dict[str, Any]:
    del output_dir
    result = collect(config, shared_state)
    return {
        "status": "completed",
        "result": result,
        "record": result["record"],
        "child_stem": "experiment-a-vanilla-profiling",
    }
