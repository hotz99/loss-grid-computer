from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias


# ORCHESTRATOR INPUT
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    input_shape: tuple[int, ...]
    sample_count: int


@dataclass(frozen=True)
class MLTaskSpec:
    name: str
    dataset: DatasetSpec
    model: str
    task: str
    loss: str
    checkpoint_path: Optional[str] = None


@dataclass(frozen=True)
class GridSpec:
    resolution: int
    scale: float


@dataclass(frozen=True)
class VanillaMode:
    gpu_batch_size: int
    _tag: Literal["vanilla"] = "vanilla"


@dataclass(frozen=True)
class HybridMode:
    gpu_batch_size: int
    cpu_batch_size: Optional[int] = None
    cpu_workers: Optional[int] = None
    _tag: Literal["hybrid"] = "hybrid"


RunMode: TypeAlias = VanillaMode | HybridMode


@dataclass(frozen=True)
class RunRequest:
    task: MLTaskSpec
    grid: GridSpec
    mode: Optional[RunMode] = None


# SCHEDULER INPUT
@dataclass(frozen=True)
class SchedulerRequest:
    task: MLTaskSpec
    grid: GridSpec
    mode: RunMode
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"


# SCHEDULER OUTPUT
@dataclass(frozen=True)
class SurfacePoint:
    x: float
    y: float
    loss: float


@dataclass(frozen=True)
class SchedulerResult:
    mode: RunMode
    total_seconds: float
    points: tuple[SurfacePoint, ...]
