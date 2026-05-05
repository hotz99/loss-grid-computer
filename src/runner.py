from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any, Optional

import torch

from src.backends import run_backend
from src.calibration import CalibrationPolicy, run_calibration
from src.config import (
    ExperimentConfig,
    HybridExecutionConfig,
    VanillaExecutionConfig,
)
from src.results import (
    load_cached_run_summary,
    load_json,
    write_experiment_result,
    write_json,
)
from src.workloads import WORKLOADS


@dataclass(frozen=True)
class InputPayload:
    config: ExperimentConfig
    cpu_worker_values: tuple[int, ...]
    cpu_batch_sizes: tuple[int, ...]
    retry: int


@dataclass(frozen=True)
class ResolvedHardware:
    backend_class: str
    gpu_name: str
    cpu_name: str
    cpu_worker_capacity: int


@dataclass(frozen=True)
class CalibrationCacheKey:
    resolved_hardware: dict[str, Any]
    input_payload: dict[str, Any]

def run_baseline_and_persist(
    baseline_config: ExperimentConfig,
):
    result = run_backend(VanillaExecutionConfig(workload=baseline_config))
    write_experiment_result(result)
    return result


def resolve_hardware(config: ExperimentConfig) -> ResolvedHardware:
    device = torch.device(
        config.runtime.device
        if config.runtime.device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    backend_class = device.type
    if backend_class == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
    elif backend_class == "mps":
        gpu_name = "Apple MPS"
    else:
        gpu_name = "none"
    cpu_name = platform.processor() or platform.machine() or "unknown-cpu"
    cpu_worker_capacity = os.cpu_count() or 1
    return ResolvedHardware(
        backend_class=backend_class,
        gpu_name=gpu_name,
        cpu_name=cpu_name,
        cpu_worker_capacity=cpu_worker_capacity,
    )


def _model_param_shape_signature(
    task: "MLTaskSpec",
) -> tuple[int, ...]:
    definition = WORKLOADS[task.name]
    model = definition.build_model(task)
    return tuple(parameter.numel() for parameter in model.parameters())


def build_calibration_cache_key(
    input_payload: InputPayload,
    resolved_hardware: ResolvedHardware,
) -> CalibrationCacheKey:
    workload_spec = input_payload.config.task
    return CalibrationCacheKey(
        resolved_hardware=asdict(resolved_hardware),
        input_payload={
            "scheduler_policy_id": "pickup_as_you_finish:v2",
            "preload_enabled": input_payload.config.runtime.preload,
            "workload_name": workload_spec.name,
            "model_family": workload_spec.model,
            "model_param_shape_signature": _model_param_shape_signature(
                input_payload.config.task,
            ),
            "task_family": workload_spec.task,
            "loss_family": workload_spec.loss,
            "dataset_family": workload_spec.dataset,
            "input_shape": workload_spec.input_shape,
            "data_batch_size": input_payload.config.data.batch_size,
            "gpu_batch_size": input_payload.config.data.gpu_batch_size,
            "gpu_slowdown_factor": input_payload.config.runtime.gpu_slowdown_factor,
            "subset_size": input_payload.config.data.subset_size,
            "grid_resolution": input_payload.config.grid.resolution,
            "num_batches": input_payload.config.runtime.num_batches,
            "cpu_worker_values": input_payload.cpu_worker_values,
            "cpu_batch_sizes": input_payload.cpu_batch_sizes,
            "retry": input_payload.retry,
        },
    )


def main(input_payload: InputPayload):
    resolved_hardware = resolve_hardware(input_payload.config)

    calibration_cache_key = build_calibration_cache_key(
        input_payload,
        resolved_hardware,
    )
    serialized_calibration_cache_key = json.dumps(
        asdict(calibration_cache_key), sort_keys=True
    )
    calibration_cache_digest = hashlib.sha256(
        serialized_calibration_cache_key.encode("utf-8")
    ).hexdigest()
    calibration_cache_path = (
        Path(input_payload.config.runtime.output_root)
        / f"{input_payload.config.experiment_name}-summary"
        / "execution_policy_cache"
        / f"{calibration_cache_digest}.json"
    )

    if calibration_cache_path.exists():
        execution_policy: CalibrationPolicy = load_json(calibration_cache_path)["value"]
    else:
        # TODO this is brittle, prone to drift
        baseline_config = input_payload.config.clone()
        try:
            baseline_summary = load_cached_run_summary(baseline_config)
        except FileNotFoundError:
            baseline_summary = run_baseline_and_persist(baseline_config).record
        baseline_total_s = baseline_summary.measurement.total_s
        execution_policy = run_calibration(
            input_payload.config,
            baseline_total_s,
            input_payload.cpu_worker_values,
            input_payload.cpu_batch_sizes,
            input_payload.retry,
        )
        write_json(
            calibration_cache_path,
            {
                "key": asdict(calibration_cache_key),
                "value": execution_policy,
            },
        )

    execution_config = (
        VanillaExecutionConfig(workload=input_payload.config)
        if execution_policy["_tag"] == "baseline"
        else HybridExecutionConfig(
            workload=input_payload.config,
            cpu_workers=int(execution_policy["cpu"]["workers"]),
            cpu_batch_size=int(execution_policy["cpu"]["batch_size"]),
        )
    )
    execution_result = run_backend(execution_config)
    write_experiment_result(execution_result)
