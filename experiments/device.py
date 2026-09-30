from __future__ import annotations

import random
import time
from typing import Literal

import torch


DeviceName = Literal["auto", "cpu", "mps", "cuda"]


def resolve(name: DeviceName | str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        mps_sync = getattr(torch.mps, "synchronize", None)
        if mps_sync is not None:
            mps_sync()


def seed_all(device: torch.device, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.type == "mps" and hasattr(torch, "mps"):
        mps_seed = getattr(torch.mps, "manual_seed", None)
        if mps_seed is not None:
            mps_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def apply_gpu_slowdown(
    device: torch.device,
    gpu_slowdown_factor: float,
    elapsed_s: float,
) -> None:
    # Sensitivity-sweep instrument: software analog of the DVFS throughput
    # parameter [mei2017gpudvfs], used to sweep Gallet's r across regimes the
    # native workload set does not span.
    if device.type == "cpu" or gpu_slowdown_factor <= 1.0:
        return
    synchronize(device)
    extra_delay = elapsed_s * (gpu_slowdown_factor - 1.0)
    if extra_delay > 0:
        time.sleep(extra_delay)
