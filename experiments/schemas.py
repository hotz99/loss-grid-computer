from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias


DEFAULT_WORKLOADS: tuple[str, ...] = (
    "cifar10_resnet20_classification",
    "cifar10_row_gru_classification",
    "california_mlp_regression",
    "mnist_mlp_classification",
)

Surface: TypeAlias = tuple[tuple[int, int, float], ...]
SurfaceValidation: TypeAlias = dict[str, Any]


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
    checkpoint_path: str | None = None


@dataclass(frozen=True)
class GridSpec:
    resolution: int
    scale: float


@dataclass(frozen=True)
class SurfaceGateConfig:
    rel_tol: float = 1e-5
    abs_tol: float = 0.0


@dataclass(frozen=True)
class Experiment1Config:
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    workload_names: tuple[str, ...] = DEFAULT_WORKLOADS
    seed: int = 1337
    sample_count: int = 1024
    grid: GridSpec = field(default_factory=lambda: GridSpec(8, 1.0))
    batch_size: int = 64
    repeats: int = 5
    point_chunk_sizes: tuple[int, ...] = (32, 64)
    max_memory_fraction: float | None = 0.85
    include_compile_candidates: bool = False
    surface_gate: SurfaceGateConfig = field(default_factory=SurfaceGateConfig)


@dataclass(frozen=True)
class Experiment2Config:
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    workload_names: tuple[str, ...] = DEFAULT_WORKLOADS
    seed: int = 1337
    sample_count: int = 1024
    grid: GridSpec = field(default_factory=lambda: GridSpec(8, 1.0))
    gpu_batch_size: int = 64
    repeats: int = 5
    # Base-2 geometric slowdown ladder {1, 2, 4, 8, 16} swept in full from
    # slow=1; the ceiling caps the highest rung. r_native does not gate it.
    slowdown_ceiling: int = 16
    max_cpu_worker_candidate: int | None = None
    surface_gate: SurfaceGateConfig = field(default_factory=SurfaceGateConfig)


@dataclass(frozen=True)
class Experiment3Config:
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    seed: int = 1337
    sample_count: int = 1024
    session_grid: GridSpec = field(default_factory=lambda: GridSpec(20, 1.0))
    gpu_batch_size: int = 64
    composition_selection_patience: int = 3
    max_cpu_worker_candidate: int | None = None
    surface_gate: SurfaceGateConfig = field(default_factory=SurfaceGateConfig)


@dataclass(frozen=True)
class TrialSpec:
    experiment: str
    workload_name: str
    candidate: str
    repeat: int
    trial_order: tuple[str, ...] = ()
    control: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateRunResult:
    workload_name: str
    candidate: str
    role: str
    repeat: int
    status: str
    trial_order: tuple[str, ...]
    total_grid_s: float | None = None
    records: Surface = ()
    validation: SurfaceValidation | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class CandidateAggregate:
    workload_name: str
    candidate: str
    role: str
    speedup_mean: float | None
    speedup_ci_low: float | None
    speedup_ci_high: float | None
    claim_status: str
    repeats: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Experiment1Result:
    status: str
    schema_version: str
    config: Experiment1Config
    trials: tuple[TrialSpec, ...]
    runs: tuple[CandidateRunResult, ...]
    aggregates: tuple[CandidateAggregate, ...]
    rq3_config: str
    composition: dict[str, Any]
    record: dict[str, Any]


@dataclass(frozen=True)
class Experiment2Result:
    status: str
    schema_version: str
    config: Experiment2Config
    result: dict[str, Any]
    record: dict[str, Any]


@dataclass(frozen=True)
class Experiment3Result:
    status: str
    schema_version: str
    config: Experiment3Config
    result: dict[str, Any]
    record: dict[str, Any]
