from __future__ import annotations

from dataclasses import asdict
import time

from torch.nn.utils import vector_to_parameters

from src.backends.base import (
    apply_gpu_slowdown,
    build_grid_points,
    build_output_dir,
    evaluate_points_on_device,
    prepare_model_and_data,
    resolve_device,
    throughput,
)
from src.system_schema import SchedulerRequest, VanillaMode
from src.results import (
    DeviceRecord,
    ExperimentResult,
    Measurement,
    RunRecord,
    synchronize_device,
)


def run(
    request: SchedulerRequest,
    seed: int = 1337,
    gpu_slowdown_factor: float = 1.0,
):
    assert isinstance(request.mode, VanillaMode)
    torch_device = resolve_device(request.device)
    (
        model,
        data_loader,
        preload_s,
        base_vector_cpu,
        direction_a_cpu,
        direction_b_cpu,
    ) = prepare_model_and_data(
        request,
        torch_device,
        seed,
    )

    points = build_grid_points(request.grid)

    base_vector_device = base_vector_cpu.to(torch_device)
    direction_a_device = direction_a_cpu.to(torch_device)
    direction_b_device = direction_b_cpu.to(torch_device)
    vector_to_parameters(base_vector_device, model.parameters())
    synchronize_device(torch_device)

    synchronize_device(torch_device)
    total_start = time.perf_counter()
    surface_records = evaluate_points_on_device(
        request,
        model,
        data_loader,
        torch_device,
        points,
        base_vector_device,
        direction_a_device,
        direction_b_device,
    )
    synchronize_device(torch_device)
    evaluation_runtime = time.perf_counter() - total_start
    apply_gpu_slowdown(
        torch_device,
        gpu_slowdown_factor,
        evaluation_runtime,
    )
    synchronize_device(torch_device)
    total_runtime = time.perf_counter() - total_start
    total_points = len(points)
    total_throughput = throughput(total_points, total_runtime)

    print(
        f"[vanilla] device={torch_device.type} "
        f"grid_time={total_runtime:.4f}s "
        f"throughput={total_throughput:.4f}pts/s "
        f"points={total_points}"
    )

    return ExperimentResult(
        record=RunRecord(
            experiment_name=request.task.name,
            measurement=Measurement(
                total_s=float(total_runtime),
                num_points=total_points,
            ),
            backend=request.mode._tag,
            device=DeviceRecord(str(torch_device), 0),
            config=asdict(request),
            comparison=None,
            output_dir=build_output_dir(request),
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
        records=surface_records,
    )
