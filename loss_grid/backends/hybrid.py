from __future__ import annotations

from dataclasses import asdict
import multiprocessing as mp
import os
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


def _empty_stage_breakdown_dict() -> Dict[str, float]:
    return asdict(StageBreakdown())


def _apply_worker_nice(config_dict: Dict) -> None:
    nice_delta = int(config_dict.get("runtime", {}).get("cpu_worker_nice", 0))
    if nice_delta <= 0:
        return
    try:
        os.nice(nice_delta)
    except OSError:
        pass


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

    profiler.section_start("chunk_point_evaluation")
    records = executor._evaluate_points_on_device(
        context=context,
        points=points,
        base_vector_device=base_vector_device,
        direction_a_device=direction_a_device,
        direction_b_device=direction_b_device,
        stage_breakdown=stage_breakdown,
    )
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
    _apply_worker_nice(config_dict)
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
    idle_pct = (
        ((total_wall_s - total_eval_s) / total_wall_s * 100) if total_wall_s > 0 else 0
    )

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
    _apply_worker_nice(config_dict)
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


def _device_worker_static_loop(
    config_dict: Dict,
    points: Sequence[GridPoint],
    result_queue,
    worker_id: int,
    device_override: str,
    worker_type: str,
    worker_label: str,
) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _apply_worker_nice(config_dict)
    if device_override == "cpu":
        torch.set_num_threads(
            max(1, int(config_dict["decomposition"]["cpu_threads_per_worker"]))
        )
    setup_start = time.perf_counter()
    config = experiment_config_from_dict(config_dict)
    context = build_execution_context(
        config, device_override=device_override, capture_env_info=False
    )
    setup_s = time.perf_counter() - setup_start
    executor = _ChunkExecutor()
    stage_breakdown = StageBreakdown()
    wall_start = time.perf_counter()
    records = _evaluate_chunk(executor, context, points, stage_breakdown)
    wall_s = time.perf_counter() - wall_start

    print(
        f"[{worker_label}] wall={wall_s:.4f}s "
        f"eval={wall_s:.4f}s "
        f"lock_wait=0.0000s "
        f"idle=0.0% "
        f"chunks={1 if points else 0}"
    )

    result_queue.put(
        {
            "worker_type": worker_type,
            "worker_id": worker_id,
            "points_processed": len(points),
            "records": records,
            "stage_breakdown": asdict(stage_breakdown),
            "wall_s": wall_s,
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
                "gpu_initial_ratio": 1.0,
                "gpu_initial_points": len(points),
                "cpu_remaining_points": 0,
                "cpu_helpers_enabled": False,
                "gpu_points_processed": len(points),
                "cpu_points_processed": 0,
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
            profiler = get_profiler()
            profiler.snapshot("cpu_only_mode_start")

            if cpu_schedule == "static":
                worker_count = max(1, cpu_workers)
                partitions = [
                    points[index::worker_count] for index in range(worker_count)
                ]
                ctx = mp.get_context("spawn")
                result_queue = ctx.Queue()
                workers = []
                spawn_start = time.perf_counter()
                for worker_id in range(worker_count):
                    process = ctx.Process(
                        target=_device_worker_static_loop,
                        args=(
                            config_dict,
                            partitions[worker_id],
                            result_queue,
                            worker_id,
                            "cpu",
                            "cpu",
                            f"cpu_worker_{worker_id}",
                        ),
                    )
                    process.start()
                    workers.append(process)
                worker_spawn_wall_s = time.perf_counter() - spawn_start
                total_start = time.perf_counter()
                try:
                    for _ in range(worker_count):
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
                        stage_breakdown.transfer_s += local["transfer_s"]
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
                worker_count = max(1, cpu_workers)
                spawn_start = time.perf_counter()
                for worker_id in range(worker_count):
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
                worker_spawn_wall_s = time.perf_counter() - spawn_start
                total_start = time.perf_counter()
                try:
                    for _ in range(worker_count):
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
                        stage_breakdown.transfer_s += local["transfer_s"]
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
                "effective_cpu_lanes": max(1, cpu_workers),
                "cpu_schedule": cpu_schedule,
                "cpu_chunk_size": cpu_chunk_size,
                "cpu_helpers_enabled": cpu_workers > 0,
                "cpu_points_processed": cpu_points,
                "cpu_worker_points_processed": cpu_worker_points,
                "cpu_worker_wall_s": cpu_worker_wall_s,
                "cpu_worker_wall_s_total": sum(cpu_worker_wall_s.values()),
                "cpu_worker_wall_s_max": max(cpu_worker_wall_s.values(), default=0.0),
                "worker_spawn_wall_s": worker_spawn_wall_s,
            }
            return result

        cpu_chunk_size = max(1, config.decomposition.cpu_chunk_size)
        fixed_gpu_chunk_size = config.decomposition.fixed_gpu_chunk_size
        gpu_initial_ratio = float(config.decomposition.gpu_initial_ratio)
        gpu_initial_ratio = max(0.0, min(1.0, gpu_initial_ratio))
        if fixed_gpu_chunk_size is not None:
            gpu_chunk_size = max(
                1,
                min(
                    len(points),
                    config.decomposition.gpu_chunk_size_max,
                    int(fixed_gpu_chunk_size),
                ),
            )
        else:
            gpu_chunk_size = max(
                1, min(len(points), int(round(len(points) * gpu_initial_ratio)))
            )

        print(
            "[hybrid] allocation "
            f"total_points={len(points)} "
            f"gpu_initial_ratio={gpu_initial_ratio:.3f} "
            f"fixed_gpu_chunk_size={fixed_gpu_chunk_size} "
            f"gpu_initial_points={gpu_chunk_size} "
            f"cpu_chunk_size={cpu_chunk_size}"
        )

        surface = self._make_surface(config.grid.resolution)
        stage_breakdown = StageBreakdown()
        cpu_points = 0
        cpu_worker_points = {}
        cpu_worker_wall_s = {}
        cpu_worker_setup_s = {}
        cpu_worker_first_claim_offset_s = {}

        gpu_points_subset = list(points[:gpu_chunk_size])
        remaining_points = list(points[gpu_chunk_size:])
        cpu_helpers_enabled = bool(remaining_points) and cpu_workers > 0
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        workers = []
        config_dict = config.to_dict()
        spawn_start = time.perf_counter()
        profiler = get_profiler()
        profiler.snapshot("hybrid_spawn_start")

        if cpu_helpers_enabled:
            next_index = ctx.Value("i", 0)
            lock = ctx.Lock()
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

        gpu_phase_wall_s = 0.0
        result_collect_wall_s = 0.0
        gpu_points = 0
        gpu_worker_wall_s = 0.0

        try:
            total_start = time.perf_counter()
            gpu_records = _evaluate_chunk(
                self,
                context,
                gpu_points_subset,
                stage_breakdown,
            )
            gpu_worker_wall_s = time.perf_counter() - total_start
            gpu_phase_wall_s = gpu_worker_wall_s
            gpu_points = len(gpu_points_subset)
            for row, col, value in gpu_records:
                surface[row, col] = value

            print(
                f"[gpu_worker] wall={gpu_worker_wall_s:.4f}s "
                f"eval={gpu_worker_wall_s:.4f}s "
                f"lock_wait=0.0000s "
                f"idle=0.0% "
                f"chunks={1 if gpu_points_subset else 0}"
            )

            result_collect_start = time.perf_counter()
            profiler.snapshot("cpu_result_collect_start")
            expected_messages = cpu_workers if cpu_helpers_enabled else 0
            for _ in range(expected_messages):
                payload = result_queue.get()
                cpu_points += payload["points_processed"]
                cpu_worker_points[f"worker_{payload['worker_id']}"] = payload[
                    "points_processed"
                ]
                cpu_worker_wall_s[f"worker_{payload['worker_id']}"] = payload["wall_s"]
                cpu_worker_setup_s[f"worker_{payload['worker_id']}"] = payload.get(
                    "setup_s"
                )
                cpu_worker_first_claim_offset_s[f"worker_{payload['worker_id']}"] = (
                    payload.get("first_claim_offset_s")
                )
                for row, col, value in payload["records"]:
                    surface[row, col] = value
                local = payload["stage_breakdown"]
                stage_breakdown.transfer_s += local["transfer_s"]
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
            "mode": "hybrid_hetero",
            "cpu_workers": cpu_workers,
            "cpu_chunk_size": cpu_chunk_size,
            "gpu_chunk_size": gpu_chunk_size,
            "cpu_helpers_enabled": cpu_helpers_enabled,
            "gpu_initial_ratio": gpu_initial_ratio,
            "gpu_initial_points": len(gpu_points_subset),
            "cpu_remaining_points": len(remaining_points),
            "gpu_points_processed": gpu_points,
            "cpu_points_processed": cpu_points,
            "cpu_worker_points_processed": cpu_worker_points,
            "gpu_worker_wall_s": gpu_worker_wall_s,
            "cpu_worker_wall_s": cpu_worker_wall_s,
            "cpu_worker_setup_s": cpu_worker_setup_s,
            "cpu_worker_first_claim_offset_s": cpu_worker_first_claim_offset_s,
            "cpu_worker_wall_s_total": sum(cpu_worker_wall_s.values()),
            "cpu_worker_wall_s_max": max(cpu_worker_wall_s.values(), default=0.0),
            "worker_spawn_wall_s": worker_spawn_wall_s,
            "gpu_phase_wall_s": gpu_phase_wall_s,
            "result_collect_wall_s": result_collect_wall_s,
        }
        return result

    def _run_queue_worker(
        self,
        context,
        points,
        next_index,
        lock,
        chunk_size: int,
        worker_label: str = "gpu_worker",
        claim_section: str = "gpu_queue_claim",
        eval_section: str = "gpu_chunk_eval",
    ):
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
            profiler.section_start(claim_section)
            start, end = _claim_chunk(
                next_index=next_index,
                lock=lock,
                chunk_size=chunk_size,
                total_points=total_points,
            )
            profiler.section_end(claim_section)
            claim_elapsed = time.perf_counter() - claim_start
            total_lock_wait_s += claim_elapsed

            if start >= end:
                break
            chunk = points[start:end]

            eval_start = time.perf_counter()
            profiler.section_start(eval_section)
            records = _evaluate_chunk(
                self,
                context,
                chunk,
                stage_breakdown,
            )
            local_records.extend(records)
            profiler.section_end(eval_section)
            eval_elapsed = time.perf_counter() - eval_start
            total_eval_s += eval_elapsed

            claimed_points += len(chunk)
            chunk_count += 1

        total_wall_s = time.perf_counter() - wall_start
        idle_pct = (
            ((total_wall_s - total_eval_s) / total_wall_s * 100)
            if total_wall_s > 0
            else 0
        )

        print(
            f"[{worker_label}] wall={total_wall_s:.4f}s "
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
