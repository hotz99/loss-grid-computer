from __future__ import annotations

import time

import torch
from torch.nn.utils import vector_to_parameters

from src.backends.base import (
    apply_gpu_slowdown,
    build_grid_points,
    build_output_dir,
    evaluate_points_on_device,
    prepare_model_and_data,
    throughput,
)
from src.config import VanillaExecutionConfig
from src.results import (
    DeviceRecord,
    ExperimentResult,
    Measurement,
    RunRecord,
    synchronize_device,
)


def run(config: VanillaExecutionConfig):
    workload = config.workload
    device = torch.device(
        workload.runtime.device
        if workload.runtime.device != "auto"
        else (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    )
    (
        model,
        data_loader,
        preload_s,
        base_vector_cpu,
        direction_a_cpu,
        direction_b_cpu,
    ) = prepare_model_and_data(
        config,
        device,
    )

    points = build_grid_points(workload.grid)

    base_vector_device = base_vector_cpu.to(device)
    direction_a_device = direction_a_cpu.to(device)
    direction_b_device = direction_b_cpu.to(device)
    vector_to_parameters(base_vector_device, model.parameters())
    synchronize_device(device)

    synchronize_device(device)
    total_start = time.perf_counter()
    records = evaluate_points_on_device(
        config=config,
        model=model,
        data_loader=data_loader,
        device=device,
        chunk=points,
        base_vector_device=base_vector_device,
        direction_a_device=direction_a_device,
        direction_b_device=direction_b_device,
    )
    synchronize_device(device)
    evaluation_runtime = time.perf_counter() - total_start
    apply_gpu_slowdown(
        device=device,
        gpu_slowdown_factor=workload.runtime.gpu_slowdown_factor,
        elapsed_s=evaluation_runtime,
    )
    synchronize_device(device)
    total_runtime = time.perf_counter() - total_start
    total_points = len(points)
    total_throughput = throughput(total_points, total_runtime)

    print(
        f"[vanilla] device={device.type} "
        f"grid_time={total_runtime:.4f}s "
        f"throughput={total_throughput:.4f}pts/s "
        f"points={total_points}"
    )

    return ExperimentResult(
        record=RunRecord(
            experiment_name=workload.experiment_name,
            measurement=Measurement(
                total_s=float(total_runtime),
                num_points=total_points,
            ),
            backend=config._tag,
            device=DeviceRecord(str(device), 0),
            config=workload,
            comparison=None,
            output_dir=build_output_dir(config),
        ),
        runtime_log={
            "total_s": total_runtime,
            "vanilla_execution": {
                "grid_compute_only_s": total_runtime,
                "points_processed": total_points,
                "throughput_points_per_s": total_throughput,
                "preload_s": preload_s,
            },
        },
        records=records,
    )
