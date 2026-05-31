from __future__ import annotations

from dataclasses import dataclass

import torch

from experiments import device as device_mod
from experiments.candidates import baseline
from experiments.schemas import GridSpec, MLTaskSpec


@dataclass(frozen=True)
class ThroughputProbe:
    cpu_throughput_pts_s: float
    gpu_throughput_pts_s: float
    r_native: float
    cpu_total_s: float
    gpu_total_s: float
    probe_grid_points: int


def _throughput(grid_points: int, total_s: float) -> float:
    return float(grid_points) / max(total_s, 1e-9)


def measure(
    task: MLTaskSpec,
    grid: GridSpec,
    *,
    gpu_batch_size: int,
    cpu_batch_size: int,
    gpu_device: torch.device,
    seed: int,
) -> ThroughputProbe:
    cpu_device = torch.device("cpu")
    cpu_result = baseline.run(
        task, grid,
        batch_size=cpu_batch_size, device=cpu_device, seed=seed,
        gpu_slowdown_factor=1.0,
    )
    gpu_result = baseline.run(
        task, grid,
        batch_size=gpu_batch_size, device=gpu_device, seed=seed,
        gpu_slowdown_factor=1.0,
    )
    grid_points = grid.resolution * grid.resolution
    cpu_pts_s = _throughput(grid_points, cpu_result.total_grid_s)
    gpu_pts_s = _throughput(grid_points, gpu_result.total_grid_s)
    r_native = cpu_pts_s / max(gpu_pts_s, 1e-9)
    device_mod.synchronize(gpu_device)
    return ThroughputProbe(
        cpu_throughput_pts_s=cpu_pts_s,
        gpu_throughput_pts_s=gpu_pts_s,
        r_native=r_native,
        cpu_total_s=cpu_result.total_grid_s,
        gpu_total_s=gpu_result.total_grid_s,
        probe_grid_points=grid_points,
    )
