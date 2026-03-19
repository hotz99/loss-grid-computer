from __future__ import annotations

from dataclasses import asdict
import multiprocessing as mp
import signal
import time
from typing import Dict, List, Sequence, Tuple

import torch

from loss_grid.backends.base import BaseLossGridExecutor
from loss_grid.config import ExperimentConfig, experiment_config_from_dict
from loss_grid.grid import GridPoint
from loss_grid.instrumentation import StageBreakdown
from loss_grid.kernel import build_execution_context
from loss_grid.profiling import get_profiler


def _claim_chunk(
    next_index,
    lock,
    chunk_size: int,
    total_points: int,
) -> Tuple[int, int]:
    profiler = get_profiler()
    profiler.section_start("queue_lock_wait")
    with lock:
        profiler.section_end("queue_lock_wait")
        profiler.section_start("queue_claim_update")
        start = next_index.value
        if start >= total_points:
            profiler.section_end("queue_claim_update")
            return total_points, total_points
        end = min(total_points, start + max(1, chunk_size))
        next_index.value = end
        profiler.section_end("queue_claim_update")
        return start, end


def _evaluate_chunk(
    executor: BaseLossGridExecutor,
    context,
    points: Sequence[GridPoint],
    stage_breakdown: StageBreakdown,
) -> List[Tuple[int, int, float]]:
    profiler = get_profiler()
    profiler.section_start("chunk_data_transfer")
    base_vector_device = context.base_vector_cpu.to(context.device)
    direction_a_device = context.direction_a_cpu.to(context.device)
    direction_b_device = context.direction_b_cpu.to(context.device)
    profiler.section_end("chunk_data_transfer")
    
    records = []
    profiler.section_start("chunk_point_evaluation")
    for point in points:
        loss_value = executor._evaluate_point_on_device(
            context=context,
            alpha=point.alpha,
            beta=point.beta,
            base_vector_device=base_vector_device,
            direction_a_device=direction_a_device,
            direction_b_device=direction_b_device,
            stage_breakdown=stage_breakdown,
        )
        records.append((point.row, point.col, loss_value))
    profiler.section_end("chunk_point_evaluation")
    return records


def _cpu_worker_loop(
    config_dict: Dict,
    points: Sequence[GridPoint],
    next_index,
    lock,
    result_queue,
    worker_id: int,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    setup_start = time.perf_counter()
    torch.set_num_threads(
        max(1, int(config_dict["decomposition"]["cpu_threads_per_worker"]))
    )
    config = experiment_config_from_dict(config_dict)
    context = build_execution_context(
        config, device_override="cpu", capture_env_info=False
    )
    setup_s = time.perf_counter() - setup_start
    executor = _ChunkExecutor()
    stage_breakdown = StageBreakdown()
    local_records = []
    claimed_points = 0
    wall_start = time.perf_counter()
    first_claim_offset_s = None
    total_lock_wait_s = 0.0
    total_eval_s = 0.0
    chunk_count = 0

    while True:
        claim_start = time.perf_counter()
        start, end = _claim_chunk(
            next_index=next_index,
            lock=lock,
            chunk_size=config.decomposition.cpu_chunk_size,
            total_points=len(points),
        )
        claim_elapsed = time.perf_counter() - claim_start
        total_lock_wait_s += claim_elapsed
        
        if start >= end:
            break
        if first_claim_offset_s is None:
            first_claim_offset_s = time.perf_counter() - wall_start
        chunk = points[start:end]
        
        eval_start = time.perf_counter()
        local_records.extend(_evaluate_chunk(executor, context, chunk, stage_breakdown))
        eval_elapsed = time.perf_counter() - eval_start
        total_eval_s += eval_elapsed
        
        claimed_points += len(chunk)
        chunk_count += 1

    total_wall_s = time.perf_counter() - wall_start
    idle_pct = ((total_wall_s - total_eval_s) / total_wall_s * 100) if total_wall_s > 0 else 0
    
    print(
        f"[cpu_worker_{worker_id}] wall={total_wall_s:.4f}s "
        f"eval={total_eval_s:.4f}s "
        f"lock_wait={total_lock_wait_s:.4f}s "
        f"idle={idle_pct:.1f}% "
        f"chunks={chunk_count}"
    )

    result_queue.put(
        {
            "worker_type": "cpu",
            "worker_id": worker_id,
            "points_processed": claimed_points,
            "records": local_records,
            "stage_breakdown": asdict(stage_breakdown),
            "wall_s": time.perf_counter() - wall_start,
            "setup_s": setup_s,
            "first_claim_offset_s": first_claim_offset_s,
        }
    )


def _cpu_worker_static_loop(
    config_dict: Dict,
    points: Sequence[GridPoint],
    result_queue,
    worker_id: int,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    setup_start = time.perf_counter()
    torch.set_num_threads(
        max(1, int(config_dict["decomposition"]["cpu_threads_per_worker"]))
    )
    config = experiment_config_from_dict(config_dict)
    context = build_execution_context(
        config, device_override="cpu", capture_env_info=False
    )
    setup_s = time.perf_counter() - setup_start
    executor = _ChunkExecutor()
    stage_breakdown = StageBreakdown()
    wall_start = time.perf_counter()
    records = _evaluate_chunk(executor, context, points, stage_breakdown)
    result_queue.put(
        {
            "worker_type": "cpu",
            "worker_id": worker_id,
            "points_processed": len(points),
            "records": records,
            "stage_breakdown": asdict(stage_breakdown),
            "wall_s": time.perf_counter() - wall_start,
            "setup_s": setup_s,
            "first_claim_offset_s": 0.0 if points else None,
        }
    )


class _ChunkExecutor(BaseLossGridExecutor):
    def run(self, config: ExperimentConfig):
        raise NotImplementedError


class HybridLossGridExecutor(BaseLossGridExecutor):
    def run(self, config: ExperimentConfig):
        context, surface = self._common_setup(config)
        output_dir = self._output_dir(config)
        cpu_workers = max(0, config.resources.cpu_workers)
        points = self._partition(config, context.points, rank=0, workers=1)

        if cpu_workers == 0:
            stage_breakdown = StageBreakdown()
            base_vector_device = context.base_vector_cpu.to(context.device)
            direction_a_device = context.direction_a_cpu.to(context.device)
            direction_b_device = context.direction_b_cpu.to(context.device)
            total_start = time.perf_counter()
            eval_start = time.perf_counter()
            for point in points:
                surface[point.row, point.col] = self._evaluate_point_on_device(
                    context=context,
                    alpha=point.alpha,
                    beta=point.beta,
                    base_vector_device=base_vector_device,
                    direction_a_device=direction_a_device,
                    direction_b_device=direction_b_device,
                    stage_breakdown=stage_breakdown,
                )
            eval_elapsed = time.perf_counter() - eval_start
            total_runtime = time.perf_counter() - total_start
            
            print(
                f"[gpu_only] wall={total_runtime:.4f}s "
                f"eval={eval_elapsed:.4f}s "
                f"points={len(points)}"
            )
            
            stage_breakdown.finalize(total_runtime)
            result = self._finalize_result(
                config=config,
                surface=surface,
                stage_breakdown=stage_breakdown,
                environment=context.environment,
                device_name=str(context.device),
                rank=0,
                world_size=1,
                output_dir=output_dir,
                is_root=True,
            )
            result.runtime_log["hybrid_scheduler"] = {
                "mode": "hybrid_hetero",
                "cpu_workers": 0,
                "cpu_chunk_size": 0,
                "gpu_chunk_size": len(points),
                "gpu_calibration_points": 0,
                "cpu_calibration_points": 0,
                "estimated_gpu_points_per_s": None,
                "estimated_cpu_points_per_s_per_worker": None,
                "gpu_points_processed_after_calibration": len(points),
                "cpu_points_processed_after_calibration": 0,
            }
            return result

        if context.device.type == "cpu":
            cpu_chunk_size = max(1, config.decomposition.cpu_chunk_size)
            cpu_schedule = config.decomposition.cpu_schedule.lower()
            stage_breakdown = StageBreakdown()
            cpu_points = 0
            cpu_worker_points = {}
            cpu_worker_wall_s = {}

            config_dict = config.to_dict()
            total_start = time.perf_counter()
            profiler = get_profiler()
            profiler.snapshot("cpu_only_mode_start")
            
            if cpu_schedule == "static":
                worker_count = cpu_workers + 1
                partitions = [
                    points[index::worker_count] for index in range(worker_count)
                ]
                ctx = mp.get_context("spawn")
                result_queue = ctx.Queue()
                workers = []
                for worker_id in range(cpu_workers):
                    process = ctx.Process(
                        target=_cpu_worker_static_loop,
                        args=(
                            config_dict,
                            partitions[worker_id + 1],
                            result_queue,
                            worker_id,
                        ),
                    )
                    process.start()
                    workers.append(process)
                try:
                    gpu_stage = StageBreakdown()
                    coordinator_start = time.perf_counter()
                    coordinator_records = _evaluate_chunk(
                        self, context, partitions[0], gpu_stage
                    )
                    gpu_worker_wall_s = time.perf_counter() - coordinator_start
                    gpu_points = len(partitions[0])
                    for row, col, value in coordinator_records:
                        surface[row, col] = value
                    stage_breakdown.perturbation_s += gpu_stage.perturbation_s
                    stage_breakdown.transfer_s += gpu_stage.transfer_s
                    stage_breakdown.forward_s += gpu_stage.forward_s
                    stage_breakdown.communication_s += gpu_stage.communication_s
                    stage_breakdown.gpu_kernel_s += gpu_stage.gpu_kernel_s
                    stage_breakdown.host_preprocessing_s += (
                        gpu_stage.host_preprocessing_s
                    )

                    for _ in range(cpu_workers):
                        payload = result_queue.get()
                        cpu_points += payload["points_processed"]
                        cpu_worker_points[f"worker_{payload['worker_id']}"] = payload[
                            "points_processed"
                        ]
                        cpu_worker_wall_s[f"worker_{payload['worker_id']}"] = payload[
                            "wall_s"
                        ]
                        for row, col, value in payload["records"]:
                            surface[row, col] = value
                        local = payload["stage_breakdown"]
                        stage_breakdown.perturbation_s += local["perturbation_s"]
                        stage_breakdown.transfer_s += local["transfer_s"]
                        stage_breakdown.forward_s += local["forward_s"]
                        stage_breakdown.communication_s += local["communication_s"]
                        stage_breakdown.gpu_kernel_s += local["gpu_kernel_s"]
                        stage_breakdown.host_preprocessing_s += local[
                            "host_preprocessing_s"
                        ]
                except KeyboardInterrupt:
                    for process in workers:
                        if process.is_alive():
                            process.terminate()
                    for process in workers:
                        process.join()
                    raise
                else:
                    for process in workers:
                        process.join()
            else:
                ctx = mp.get_context("spawn")
                next_index = ctx.Value("i", 0)
                lock = ctx.Lock()
                result_queue = ctx.Queue()
                workers = []
                for worker_id in range(cpu_workers):
                    process = ctx.Process(
                        target=_cpu_worker_loop,
                        args=(
                            config_dict,
                            points,
                            next_index,
                            lock,
                            result_queue,
                            worker_id,
                        ),
                    )
                    process.start()
                    workers.append(process)
                try:
                    gpu_records, gpu_breakdown, gpu_points, gpu_worker_wall_s = (
                        self._run_gpu_worker(
                            context=context,
                            points=points,
                            next_index=next_index,
                            lock=lock,
                            chunk_size=cpu_chunk_size,
                        )
                    )
                    for row, col, value in gpu_records:
                        surface[row, col] = value
                    stage_breakdown.perturbation_s += gpu_breakdown.perturbation_s
                    stage_breakdown.transfer_s += gpu_breakdown.transfer_s
                    stage_breakdown.forward_s += gpu_breakdown.forward_s
                    stage_breakdown.communication_s += gpu_breakdown.communication_s
                    stage_breakdown.gpu_kernel_s += gpu_breakdown.gpu_kernel_s
                    stage_breakdown.host_preprocessing_s += (
                        gpu_breakdown.host_preprocessing_s
                    )

                    for _ in range(cpu_workers):
                        payload = result_queue.get()
                        cpu_points += payload["points_processed"]
                        cpu_worker_points[f"worker_{payload['worker_id']}"] = payload[
                            "points_processed"
                        ]
                        cpu_worker_wall_s[f"worker_{payload['worker_id']}"] = payload[
                            "wall_s"
                        ]
                        for row, col, value in payload["records"]:
                            surface[row, col] = value
                        local = payload["stage_breakdown"]
                        stage_breakdown.perturbation_s += local["perturbation_s"]
                        stage_breakdown.transfer_s += local["transfer_s"]
                        stage_breakdown.forward_s += local["forward_s"]
                        stage_breakdown.communication_s += local["communication_s"]
                        stage_breakdown.gpu_kernel_s += local["gpu_kernel_s"]
                        stage_breakdown.host_preprocessing_s += local[
                            "host_preprocessing_s"
                        ]
                except KeyboardInterrupt:
                    for process in workers:
                        if process.is_alive():
                            process.terminate()
                    for process in workers:
                        process.join()
                    raise
                else:
                    for process in workers:
                        process.join()

            total_runtime = time.perf_counter() - total_start
            stage_breakdown.finalize(total_runtime)
            result = self._finalize_result(
                config=config,
                surface=surface,
                stage_breakdown=stage_breakdown,
                environment=context.environment,
                device_name=str(context.device),
                rank=0,
                world_size=1 + cpu_workers,
                output_dir=output_dir,
                is_root=True,
            )
            result.runtime_log["hybrid_scheduler"] = {
                "mode": f"cpu_only_{cpu_schedule}",
                "cpu_workers": cpu_workers,
                "cpu_schedule": cpu_schedule,
                "cpu_chunk_size": cpu_chunk_size,
                "gpu_chunk_size": cpu_chunk_size,
                "gpu_calibration_points": 0,
                "cpu_calibration_points": 0,
                "estimated_gpu_points_per_s": None,
                "estimated_cpu_points_per_s_per_worker": None,
                "estimated_total_cpu_points_per_s": None,
                "gpu_points_processed_after_calibration": gpu_points,
                "cpu_points_processed_after_calibration": cpu_points,
                "cpu_worker_points_processed_after_calibration": cpu_worker_points,
                "gpu_worker_wall_s_after_calibration": gpu_worker_wall_s,
                "cpu_worker_wall_s_after_calibration": cpu_worker_wall_s,
                "cpu_worker_wall_s_total_after_calibration": sum(
                    cpu_worker_wall_s.values()
                ),
                "cpu_worker_wall_s_max_after_calibration": max(
                    cpu_worker_wall_s.values(), default=0.0
                ),
            }
            return result

        scheduler_info = self._calibrate_scheduler(config, points)
        calibration_count = scheduler_info["calibration_count"]
        gpu_chunk_size = scheduler_info["gpu_chunk_size"]
        cpu_chunk_size = scheduler_info["cpu_chunk_size"]
        surface = scheduler_info["surface"]
        stage_breakdown = scheduler_info["stage_breakdown"]
        cpu_points = 0
        cpu_worker_points = {}
        cpu_worker_wall_s = {}
        cpu_worker_setup_s = {}
        cpu_worker_first_claim_offset_s = {}

        remaining_points = points[calibration_count:]
        helper_policy = self._estimate_helper_policy(
            scheduler_info, len(remaining_points)
        )
        total_start = time.perf_counter()
        if remaining_points and helper_policy["enable_cpu_helpers"]:
            ctx = mp.get_context("spawn")
            next_index = ctx.Value("i", 0)
            lock = ctx.Lock()
            result_queue = ctx.Queue()
            workers = []

            config_dict = config.to_dict()
            spawn_start = time.perf_counter()
            profiler = get_profiler()
            profiler.snapshot("hybrid_spawn_start")
            
            for worker_id in range(cpu_workers):
                process = ctx.Process(
                    target=_cpu_worker_loop,
                    args=(
                        config_dict,
                        remaining_points,
                        next_index,
                        lock,
                        result_queue,
                        worker_id,
                    ),
                )
                process.start()
                workers.append(process)
            worker_spawn_wall_s = time.perf_counter() - spawn_start
            profiler.snapshot("hybrid_spawn_complete")

            try:
                gpu_phase_start = time.perf_counter()
                profiler.snapshot("gpu_phase_start")
                gpu_records, gpu_breakdown, gpu_points, gpu_worker_wall_s = (
                    self._run_gpu_worker(
                        context=context,
                        points=remaining_points,
                        next_index=next_index,
                        lock=lock,
                        chunk_size=gpu_chunk_size,
                    )
                )
                gpu_phase_wall_s = time.perf_counter() - gpu_phase_start
                profiler.snapshot("gpu_phase_complete")
                profiler.section_start("gpu_result_assembly")
                for row, col, value in gpu_records:
                    surface[row, col] = value
                stage_breakdown.perturbation_s += gpu_breakdown.perturbation_s
                stage_breakdown.transfer_s += gpu_breakdown.transfer_s
                stage_breakdown.forward_s += gpu_breakdown.forward_s
                stage_breakdown.communication_s += gpu_breakdown.communication_s
                stage_breakdown.gpu_kernel_s += gpu_breakdown.gpu_kernel_s
                stage_breakdown.host_preprocessing_s += (
                    gpu_breakdown.host_preprocessing_s
                )
                profiler.section_end("gpu_result_assembly")

                result_collect_start = time.perf_counter()
                profiler.snapshot("cpu_result_collect_start")
                expected_cpu_messages = cpu_workers
                for _ in range(expected_cpu_messages):
                    payload = result_queue.get()
                    cpu_points += payload["points_processed"]
                    cpu_worker_points[f"worker_{payload['worker_id']}"] = payload[
                        "points_processed"
                    ]
                    cpu_worker_wall_s[f"worker_{payload['worker_id']}"] = payload[
                        "wall_s"
                    ]
                    cpu_worker_setup_s[f"worker_{payload['worker_id']}"] = payload.get(
                        "setup_s"
                    )
                    cpu_worker_first_claim_offset_s[
                        f"worker_{payload['worker_id']}"
                    ] = payload.get("first_claim_offset_s")
                    for row, col, value in payload["records"]:
                        surface[row, col] = value
                    local = payload["stage_breakdown"]
                    stage_breakdown.perturbation_s += local["perturbation_s"]
                    stage_breakdown.transfer_s += local["transfer_s"]
                    stage_breakdown.forward_s += local["forward_s"]
                    stage_breakdown.communication_s += local["communication_s"]
                    stage_breakdown.gpu_kernel_s += local["gpu_kernel_s"]
                    stage_breakdown.host_preprocessing_s += local[
                        "host_preprocessing_s"
                    ]
                result_collect_wall_s = time.perf_counter() - result_collect_start
                profiler.snapshot("cpu_result_collect_complete")
            except KeyboardInterrupt:
                for process in workers:
                    if process.is_alive():
                        process.terminate()
                for process in workers:
                    process.join()
                raise
            else:
                for process in workers:
                    process.join()

            total_runtime = scheduler_info["calibration_wall_s"] + (
                time.perf_counter() - total_start
            )
        else:
            worker_spawn_wall_s = 0.0
            gpu_phase_wall_s = 0.0
            result_collect_wall_s = 0.0
            if remaining_points:
                gpu_records = []
                gpu_breakdown = StageBreakdown()
                coordinator_start = time.perf_counter()
                base_vector_device = context.base_vector_cpu.to(context.device)
                direction_a_device = context.direction_a_cpu.to(context.device)
                direction_b_device = context.direction_b_cpu.to(context.device)
                for point in remaining_points:
                    loss_value = self._evaluate_point_on_device(
                        context=context,
                        alpha=point.alpha,
                        beta=point.beta,
                        base_vector_device=base_vector_device,
                        direction_a_device=direction_a_device,
                        direction_b_device=direction_b_device,
                        stage_breakdown=gpu_breakdown,
                    )
                    gpu_records.append((point.row, point.col, loss_value))
                gpu_worker_wall_s = time.perf_counter() - coordinator_start
                gpu_points = len(remaining_points)
                for row, col, value in gpu_records:
                    surface[row, col] = value
                stage_breakdown.perturbation_s += gpu_breakdown.perturbation_s
                stage_breakdown.transfer_s += gpu_breakdown.transfer_s
                stage_breakdown.forward_s += gpu_breakdown.forward_s
                stage_breakdown.communication_s += gpu_breakdown.communication_s
                stage_breakdown.gpu_kernel_s += gpu_breakdown.gpu_kernel_s
                stage_breakdown.host_preprocessing_s += (
                    gpu_breakdown.host_preprocessing_s
                )
                total_runtime = scheduler_info["calibration_wall_s"] + (
                    time.perf_counter() - total_start
                )
            else:
                gpu_points = scheduler_info["gpu_calibration_points"]
                gpu_worker_wall_s = 0.0
                total_runtime = scheduler_info["calibration_wall_s"]

        stage_breakdown.finalize(total_runtime)
        result = self._finalize_result(
            config=config,
            surface=surface,
            stage_breakdown=stage_breakdown,
            environment=context.environment,
            device_name=str(context.device),
            rank=0,
            world_size=1 + cpu_workers,
            output_dir=output_dir,
            is_root=True,
        )
        result.runtime_log["hybrid_scheduler"] = {
            "mode": "hybrid_hetero",
            "cpu_workers": cpu_workers,
            "cpu_chunk_size": cpu_chunk_size,
            "gpu_chunk_size": gpu_chunk_size,
            "gpu_calibration_points": scheduler_info["gpu_calibration_points"],
            "cpu_calibration_points": scheduler_info["cpu_calibration_points"],
            "estimated_gpu_points_per_s": scheduler_info["gpu_points_per_s"],
            "estimated_cpu_points_per_s_per_worker": scheduler_info["cpu_points_per_s"],
            "estimated_total_cpu_points_per_s": scheduler_info[
                "total_cpu_points_per_s"
            ],
            "cpu_helpers_enabled": helper_policy["enable_cpu_helpers"],
            "remaining_points_considered": len(remaining_points),
            "estimated_helper_overhead_s": helper_policy["estimated_helper_overhead_s"],
            "estimated_gpu_remaining_s": helper_policy["estimated_gpu_remaining_s"],
            "estimated_hybrid_remaining_s": helper_policy[
                "estimated_hybrid_remaining_s"
            ],
            "estimated_saved_s": helper_policy["estimated_saved_s"],
            "gpu_points_processed_after_calibration": gpu_points,
            "cpu_points_processed_after_calibration": cpu_points,
            "cpu_worker_points_processed_after_calibration": cpu_worker_points,
            "gpu_worker_wall_s_after_calibration": gpu_worker_wall_s,
            "cpu_worker_wall_s_after_calibration": cpu_worker_wall_s,
            "cpu_worker_setup_s": cpu_worker_setup_s,
            "cpu_worker_first_claim_offset_s": cpu_worker_first_claim_offset_s,
            "cpu_worker_wall_s_total_after_calibration": sum(
                cpu_worker_wall_s.values()
            ),
            "cpu_worker_wall_s_max_after_calibration": max(
                cpu_worker_wall_s.values(), default=0.0
            ),
            "worker_spawn_wall_s": worker_spawn_wall_s,
            "gpu_phase_wall_s": gpu_phase_wall_s,
            "result_collect_wall_s": result_collect_wall_s,
        }
        return result

    def _calibrate_scheduler(
        self, config: ExperimentConfig, points: Sequence[GridPoint]
    ) -> Dict:
        surface = self._make_surface(config.grid.resolution)
        stage_breakdown = StageBreakdown()
        calibration_points = max(1, config.decomposition.calibration_points)

        gpu_context = build_execution_context(
            config, device_override=config.runtime.device, capture_env_info=False
        )
        cpu_context = build_execution_context(
            config, device_override="cpu", capture_env_info=False
        )

        gpu_sample = list(points[:calibration_points])
        cpu_sample = list(points[calibration_points : calibration_points * 2])

        calibration_wall_start = time.perf_counter()
        gpu_stage = StageBreakdown()
        gpu_records = _evaluate_chunk(self, gpu_context, gpu_sample, gpu_stage)
        for row, col, value in gpu_records:
            surface[row, col] = value

        cpu_stage = StageBreakdown()
        cpu_records = _evaluate_chunk(self, cpu_context, cpu_sample, cpu_stage)
        for row, col, value in cpu_records:
            surface[row, col] = value

        stage_breakdown.perturbation_s += (
            gpu_stage.perturbation_s + cpu_stage.perturbation_s
        )
        stage_breakdown.transfer_s += gpu_stage.transfer_s + cpu_stage.transfer_s
        stage_breakdown.forward_s += gpu_stage.forward_s + cpu_stage.forward_s
        stage_breakdown.communication_s += (
            gpu_stage.communication_s + cpu_stage.communication_s
        )
        stage_breakdown.gpu_kernel_s += gpu_stage.gpu_kernel_s
        stage_breakdown.host_preprocessing_s += (
            gpu_stage.host_preprocessing_s + cpu_stage.host_preprocessing_s
        )

        calibration_wall_s = time.perf_counter() - calibration_wall_start
        gpu_time_per_point = self._time_per_point(gpu_stage, len(gpu_sample))
        cpu_time_per_point = self._time_per_point(cpu_stage, len(cpu_sample))
        gpu_points_per_s = 1.0 / gpu_time_per_point if gpu_time_per_point > 0 else 0.0
        cpu_points_per_s = 1.0 / cpu_time_per_point if cpu_time_per_point > 0 else 0.0
        total_cpu_points_per_s = (
            max(1, int(config.resources.cpu_workers)) * cpu_points_per_s
        )

        gpu_chunk_bias = 1.75
        fixed_gpu_chunk_size = config.decomposition.fixed_gpu_chunk_size
        if fixed_gpu_chunk_size is not None:
            gpu_chunk_size = max(
                1,
                min(
                    config.decomposition.gpu_chunk_size_max,
                    int(fixed_gpu_chunk_size),
                ),
            )
        elif gpu_points_per_s > 0.0 and total_cpu_points_per_s > 0.0:
            gpu_chunk_size = round(
                (gpu_points_per_s / total_cpu_points_per_s) * gpu_chunk_bias
            )
        else:
            gpu_chunk_size = 1
        gpu_chunk_size = max(
            1, min(config.decomposition.gpu_chunk_size_max, gpu_chunk_size)
        )

        resolved_ratio = (
            (gpu_points_per_s / total_cpu_points_per_s)
            if total_cpu_points_per_s > 0.0
            else 0.0
        )
        print(
            "[hybrid] calibration "
            f"gpu_points_per_s={gpu_points_per_s:.6f} "
            f"cpu_points_per_s_per_worker={cpu_points_per_s:.6f} "
            f"total_cpu_points_per_s={total_cpu_points_per_s:.6f} "
            f"resolved_ratio={resolved_ratio:.6f} "
            f"gpu_chunk_bias={gpu_chunk_bias:.2f} "
            f"fixed_gpu_chunk_size={fixed_gpu_chunk_size} "
            f"gpu_chunk_size={gpu_chunk_size}"
        )

        return {
            "surface": surface,
            "stage_breakdown": stage_breakdown,
            "gpu_chunk_size": gpu_chunk_size,
            "cpu_chunk_size": max(1, config.decomposition.cpu_chunk_size),
            "gpu_points_per_s": gpu_points_per_s,
            "cpu_points_per_s": cpu_points_per_s,
            "total_cpu_points_per_s": total_cpu_points_per_s,
            "calibration_count": len(gpu_sample) + len(cpu_sample),
            "gpu_calibration_points": len(gpu_sample),
            "cpu_calibration_points": len(cpu_sample),
            "calibration_wall_s": calibration_wall_s,
        }

    @staticmethod
    def _estimate_helper_policy(scheduler_info: Dict, remaining_points: int) -> Dict:
        gpu_points_per_s = float(scheduler_info["gpu_points_per_s"])
        total_cpu_points_per_s = float(scheduler_info["total_cpu_points_per_s"])
        estimated_helper_overhead_s = float(scheduler_info["calibration_wall_s"])

        if remaining_points <= 0 or gpu_points_per_s <= 0.0:
            return {
                "enable_cpu_helpers": False,
                "estimated_helper_overhead_s": estimated_helper_overhead_s,
                "estimated_gpu_remaining_s": 0.0,
                "estimated_hybrid_remaining_s": 0.0,
                "estimated_saved_s": 0.0,
            }

        estimated_gpu_remaining_s = remaining_points / gpu_points_per_s
        if total_cpu_points_per_s <= 0.0:
            estimated_hybrid_remaining_s = (
                estimated_gpu_remaining_s + estimated_helper_overhead_s
            )
            estimated_saved_s = estimated_gpu_remaining_s - estimated_hybrid_remaining_s
            enable_cpu_helpers = False
        else:
            estimated_hybrid_remaining_s = (
                remaining_points / (gpu_points_per_s + total_cpu_points_per_s)
            ) + estimated_helper_overhead_s
            estimated_saved_s = estimated_gpu_remaining_s - estimated_hybrid_remaining_s
            enable_cpu_helpers = estimated_saved_s > 0.0

        print(
            "[hybrid] policy "
            f"remaining_points={remaining_points} "
            f"gpu_only_s={estimated_gpu_remaining_s:.6f} "
            f"hybrid_s={estimated_hybrid_remaining_s:.6f} "
            f"overhead_s={estimated_helper_overhead_s:.6f} "
            f"saved_s={estimated_saved_s:.6f} "
            f"enable_cpu_helpers={enable_cpu_helpers}"
        )

        return {
            "enable_cpu_helpers": enable_cpu_helpers,
            "estimated_helper_overhead_s": estimated_helper_overhead_s,
            "estimated_gpu_remaining_s": estimated_gpu_remaining_s,
            "estimated_hybrid_remaining_s": estimated_hybrid_remaining_s,
            "estimated_saved_s": estimated_saved_s,
        }

    def _run_gpu_worker(self, context, points, next_index, lock, chunk_size: int):
        profiler = get_profiler()
        stage_breakdown = StageBreakdown()
        local_records = []
        claimed_points = 0
        total_points = len(points)
        wall_start = time.perf_counter()
        chunk_count = 0
        total_lock_wait_s = 0.0
        total_eval_s = 0.0

        while True:
            claim_start = time.perf_counter()
            profiler.section_start("gpu_queue_claim")
            start, end = _claim_chunk(
                next_index=next_index,
                lock=lock,
                chunk_size=chunk_size,
                total_points=total_points,
            )
            profiler.section_end("gpu_queue_claim")
            claim_elapsed = time.perf_counter() - claim_start
            total_lock_wait_s += claim_elapsed
            
            if start >= end:
                break
            chunk = points[start:end]
            
            eval_start = time.perf_counter()
            profiler.section_start(f"gpu_chunk_eval")
            local_records.extend(_evaluate_chunk(self, context, chunk, stage_breakdown))
            profiler.section_end(f"gpu_chunk_eval")
            eval_elapsed = time.perf_counter() - eval_start
            total_eval_s += eval_elapsed
            
            claimed_points += len(chunk)
            chunk_count += 1

        total_wall_s = time.perf_counter() - wall_start
        idle_pct = ((total_wall_s - total_eval_s) / total_wall_s * 100) if total_wall_s > 0 else 0
        
        print(
            f"[gpu_worker] wall={total_wall_s:.4f}s "
            f"eval={total_eval_s:.4f}s "
            f"lock_wait={total_lock_wait_s:.4f}s "
            f"idle={idle_pct:.1f}% "
            f"chunks={chunk_count}"
        )

        return (
            local_records,
            stage_breakdown,
            claimed_points,
            total_wall_s,
        )

    @staticmethod
    def _time_per_point(stage_breakdown: StageBreakdown, count: int) -> float:
        if count <= 0:
            return 0.0
        total = (
            stage_breakdown.perturbation_s
            + stage_breakdown.transfer_s
            + stage_breakdown.forward_s
        )
        return total / count if total > 0 else 0.0
