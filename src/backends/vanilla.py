from __future__ import annotations

from dataclasses import asdict
from time import perf_counter, perf_counter_ns
from typing import Any

import torch
from torch.nn.utils import vector_to_parameters

from src.backends.base import (
    apply_gpu_slowdown,
    backend_log,
    build_grid_points,
    build_output_dir,
    evaluate_points_on_device,
    prepare_model_and_data,
    resolve_device,
    throughput,
)
from src.schemas import SchedulerRequest, VanillaMode
from src.results import (
    DeviceRecord,
    ExperimentResult,
    Measurement,
    RunRecord,
    synchronize_device,
)
from src.workloads import WORKLOADS


def run(
    request: SchedulerRequest,
    seed: int = 1337,
    gpu_slowdown_factor: float = 1.0,
    profile_sections: bool = False,
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
    total_start = perf_counter()
    if profile_sections:
        surface_records, section_timings = _evaluate_points_with_section_profile(
            request,
            model,
            data_loader,
            torch_device,
            points,
            base_vector_device,
            direction_a_device,
            direction_b_device,
        )
    else:
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
        section_timings = None
    synchronize_device(torch_device)
    evaluation_runtime = perf_counter() - total_start
    apply_gpu_slowdown(
        torch_device,
        gpu_slowdown_factor,
        evaluation_runtime,
    )
    synchronize_device(torch_device)
    total_runtime = perf_counter() - total_start
    total_points = len(points)
    total_throughput = throughput(total_points, total_runtime)

    backend_log(
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
            "section_timings": section_timings,
        },
        records=surface_records,
    )


def _evaluate_points_with_section_profile(
    request: SchedulerRequest,
    model,
    data_loader,
    device,
    chunk,
    base_vector_device,
    direction_a_device,
    direction_b_device,
):
    records = []
    definition = WORKLOADS[request.task.name]
    timings = {
        "perturbation_s": 0.0,
        "binding_s": 0.0,
        "batch_eval_s": 0.0,
    }

    for point in chunk:
        with _section_timer(device) as elapsed:
            perturbed_variant = (
                base_vector_device
                + (point.alpha * direction_a_device)
                + (point.beta * direction_b_device)
            )
        timings["perturbation_s"] += elapsed[0]

        with _section_timer(device) as elapsed:
            vector_to_parameters(perturbed_variant, model.parameters())
            model.eval()
        timings["binding_s"] += elapsed[0]

        total_loss = 0.0
        total_examples = 0
        with _section_timer(device) as elapsed:
            with torch.no_grad():
                for batch in data_loader:
                    loss, batch_size = definition.compute_loss(model, batch, device)
                    total_loss += float(loss.cpu()) * batch_size
                    total_examples += batch_size
        timings["batch_eval_s"] += elapsed[0]

        avg_loss = total_loss / max(1, total_examples)
        records.append((point.row, point.col, avg_loss))

    return records, timings


class _section_timer:
    def __init__(self, device):
        self.device = device
        self.elapsed = [0.0]
        self.start: Any = None
        self.end: Any = None
        self.t0: int | None = None

    def __enter__(self):
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            synchronize_device(self.device)
            self.t0 = perf_counter_ns()
        return self.elapsed

    def __exit__(self, exc_type, exc, tb):
        if self.device.type == "cuda" and torch.cuda.is_available():
            self.end.record()
            torch.cuda.synchronize(self.device)
            self.elapsed[0] = self.start.elapsed_time(self.end) * 1e-3
        else:
            synchronize_device(self.device)
            self.elapsed[0] = (perf_counter_ns() - int(self.t0)) * 1e-9
        return False
