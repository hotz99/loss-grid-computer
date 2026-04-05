from __future__ import annotations

import time
from torch.nn.utils import vector_to_parameters

from src.backends.base import (
    build_output_dir,
    evaluate_points_on_device,
    make_surface,
)
from src.directions import build_parameter_vector
from src.grid import partition_points
from src.kernel import (
    build_execution_context,
    compile_chunk_evaluator,
    evaluate_loss,
)
from src.results import ExperimentResult, Measurement, RunRecord


def _evaluate_points_eager(
    context,
    points,
    base_vector_device,
    direction_a_device,
    direction_b_device,
    surface,
) -> None:
    for point in points:
        parameter_vector = build_parameter_vector(
            base_vector_device,
            direction_a_device,
            direction_b_device,
            point.alpha,
            point.beta,
        )
        vector_to_parameters(parameter_vector, context.model.parameters())
        loss_value = evaluate_loss(
            model=context.model,
            data_loader=context.data_loader,
            device=context.device,
            num_batches=context.config.runtime.num_batches,
        )
        surface[point.row, point.col] = loss_value


def _evaluate_points_compiled(
    context,
    points,
    base_vector_device,
    direction_a_device,
    direction_b_device,
    surface,
) -> None:
    for row, col, loss_value in evaluate_points_on_device(
        context=context,
        points=points,
        base_vector_device=base_vector_device,
        direction_a_device=direction_a_device,
        direction_b_device=direction_b_device,
    ):
        surface[row, col] = loss_value


def _run_single_device(config, evaluation_fn, log_gpu_only: bool) -> ExperimentResult:
    context = build_execution_context(config)
    surface = make_surface(config.grid.resolution)
    output_dir = build_output_dir(config)
    points = partition_points(
        context.points,
        config.grid,
        worker_index=0,
        worker_count=1,
    )

    base_vector_device = context.base_vector_cpu.to(context.device)
    direction_a_device = context.direction_a_cpu.to(context.device)
    direction_b_device = context.direction_b_cpu.to(context.device)

    if (
        evaluation_fn is _evaluate_points_compiled
        and context.device.type != "cpu"
        and context.compiled_gpu_chunk_eval_enabled
        and context.compiled_chunk_evaluator is None
        and isinstance(context.data_loader, list)
        and len(context.data_loader) > 0
    ):
        compile_chunk_evaluator(context)

    total_start = time.perf_counter()
    evaluation_fn(
        context,
        points,
        base_vector_device,
        direction_a_device,
        direction_b_device,
        surface,
    )
    total_runtime = time.perf_counter() - total_start

    if log_gpu_only:
        print(f"[gpu_only] wall={total_runtime:.4f}s points={len(points)}")

    result = ExperimentResult(
        record=RunRecord(
            experiment_name=config.experiment_name,
            measurement=Measurement(
                total_s=float(total_runtime),
                num_points=int(surface.numel()),
            ),
            backend=config.backend,
            device={
                "gpu": str(context.device),
                "cpu": 0,
            },
            config=config.to_dict(),
            comparison=None,
            output_dir=output_dir,
        ),
        runtime_log={"total_s": total_runtime},
        surface=surface,
    )
    if log_gpu_only:
        result.runtime_log["single_gpu"] = {
            "gpu_worker_wall_s": total_runtime,
            "gpu_points_processed": len(points),
            "preload_s": context.preload_s,
            "compile_s": context.compile_s,
        }
    return result


def run(config):
    return _run_single_device(
        config=config,
        evaluation_fn=_evaluate_points_eager,
        log_gpu_only=False,
    )


def run_compiled(config):
    return _run_single_device(
        config=config,
        evaluation_fn=_evaluate_points_compiled,
        log_gpu_only=True,
    )
