from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import torch

from loss_grid.backends.base import BaseLossGridExecutor
from loss_grid.config import ExperimentConfig
from loss_grid.instrumentation import StageBreakdown


class MpiLossGridExecutor(BaseLossGridExecutor):
    def run(self, config: ExperimentConfig):
        try:
            import numpy as np
            from mpi4py import MPI
        except ImportError as error:
            raise RuntimeError(
                "backend=mpi requires numpy and mpi4py to be installed"
            ) from error

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        world_size = comm.Get_size()

        if torch.cuda.is_available():
            torch.cuda.set_device(rank % torch.cuda.device_count())

        context, surface = self._common_setup(config)
        stage_breakdown = StageBreakdown()
        local_points = self._partition(config, context.points, rank=rank, workers=world_size)
        output_dir = self._output_dir(config) if rank == 0 else None
        output_dir = comm.bcast(output_dir, root=0)

        base_vector_device = context.base_vector_cpu.to(context.device)
        direction_a_device = context.direction_a_cpu.to(context.device)
        direction_b_device = context.direction_b_cpu.to(context.device)

        total_start = time.perf_counter()
        local_records = []
        for point in local_points:
            loss_value = self._evaluate_point_on_device(
                context=context,
                alpha=point.alpha,
                beta=point.beta,
                base_vector_device=base_vector_device,
                direction_a_device=direction_a_device,
                direction_b_device=direction_b_device,
                stage_breakdown=stage_breakdown,
            )
            local_records.append((point.row, point.col, loss_value))

        communication_start = time.perf_counter()
        communication_mode = config.mpi.communication_mode.lower()
        if communication_mode == "gather":
            gathered = comm.gather(local_records, root=0)
            if rank == 0:
                for rank_records in gathered:
                    for row, col, loss_value in rank_records:
                        surface[row, col] = loss_value
        elif communication_mode == "allreduce":
            local_surface = np.zeros((config.grid.resolution, config.grid.resolution), dtype=np.float64)
            local_mask = np.zeros((config.grid.resolution, config.grid.resolution), dtype=np.float64)
            for row, col, loss_value in local_records:
                local_surface[row, col] = loss_value
                local_mask[row, col] = 1.0
            reduced_surface = np.zeros_like(local_surface)
            reduced_mask = np.zeros_like(local_mask)
            comm.Allreduce(local_surface, reduced_surface, op=MPI.SUM)
            comm.Allreduce(local_mask, reduced_mask, op=MPI.SUM)
            reduced_mask[reduced_mask == 0.0] = 1.0
            surface = torch.from_numpy((reduced_surface / reduced_mask).astype("float32"))
        elif communication_mode == "overlap":
            surface = self._overlap_collect(
                config=config,
                comm=comm,
                mpi_module=MPI,
                rank=rank,
                world_size=world_size,
                surface=surface,
                local_records=local_records,
                all_points=context.points,
                np=np,
            )
        else:
            raise ValueError(f"Unsupported MPI communication mode: {config.mpi.communication_mode}")

        stage_breakdown.communication_s += time.perf_counter() - communication_start
        total_runtime = time.perf_counter() - total_start
        stage_breakdown.finalize(total_runtime)
        return self._finalize_result(
            config=config,
            surface=surface,
            stage_breakdown=stage_breakdown,
            environment=context.environment,
            device_name=str(context.device),
            rank=rank,
            world_size=world_size,
            output_dir=output_dir,
            is_root=(rank == 0),
        )

    def _overlap_collect(self, config, comm, mpi_module, rank, world_size, surface, local_records, all_points, np):
        chunk_size = max(1, config.mpi.overlap_chunk_size)
        chunks_by_rank = []
        total_points = self._partition_all_ranks(config, world_size, all_points)
        for rank_points in total_points:
            rank_chunk_sizes = []
            for start in range(0, len(rank_points), chunk_size):
                rank_chunk_sizes.append(min(chunk_size, len(rank_points) - start))
            chunks_by_rank.append(rank_chunk_sizes)

        if rank == 0:
            recv_requests = []
            buffers = []
            metadata = []
            for src_rank in range(1, world_size):
                for chunk_index, expected_size in enumerate(chunks_by_rank[src_rank]):
                    buffer = np.empty((expected_size, 3), dtype=np.float64)
                    request = comm.Irecv(buffer, source=src_rank, tag=chunk_index)
                    recv_requests.append(request)
                    buffers.append(buffer)
                    metadata.append((src_rank, chunk_index))

            for row, col, value in local_records:
                surface[row, col] = value

            while recv_requests:
                index, _ = mpi_module.Request.Waitany(recv_requests)
                if index == mpi_module.UNDEFINED:
                    break
                buffer = buffers.pop(index)
                recv_requests.pop(index)
                metadata.pop(index)
                for row, col, value in buffer:
                    surface[int(row), int(col)] = float(value)
            return surface

        send_requests = []
        for chunk_index, start in enumerate(range(0, len(local_records), chunk_size)):
            chunk = local_records[start : start + chunk_size]
            payload = np.asarray(chunk, dtype=np.float64)
            send_requests.append(comm.Isend(payload, dest=0, tag=chunk_index))
        if send_requests:
            mpi_module.Request.Waitall(send_requests)
        return surface

    def _partition_all_ranks(self, config: ExperimentConfig, world_size: int, all_points) -> List[List]:
        partitions = []
        for rank in range(world_size):
            partitions.append(self._partition(config, all_points, rank=rank, workers=world_size))
        return partitions
