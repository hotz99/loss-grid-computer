from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    name: str = "resnet20"
    num_classes: int = 10
    in_channels: int = 3
    image_size: List[int] = field(default_factory=lambda: [3, 32, 32])
    checkpoint_path: Optional[str] = None
    pretrained: bool = False
    hidden_dim: int = 128


@dataclass
class DataConfig:
    name: str = "cifar10"
    root: str = "assets/cifar-10-batches-py"
    split: str = "test"
    subset_size: int = 256
    batch_size: int = 32
    cpu_batch_size: Optional[int] = None
    gpu_batch_size: Optional[int] = None
    num_workers: int = 0
    num_classes: int = 10
    normalize_mean: List[float] = field(default_factory=lambda: [0.4914, 0.4822, 0.4465])
    normalize_std: List[float] = field(default_factory=lambda: [0.2470, 0.2435, 0.2616])


@dataclass
class GridConfig:
    alpha_min: float = -1.0
    alpha_max: float = 1.0
    beta_min: float = -1.0
    beta_max: float = 1.0
    resolution: int = 9


@dataclass
class RuntimeConfig:
    device: str = "auto"
    precision: str = "fp32"
    num_batches: Optional[int] = 4
    preload_gpu_batches: bool = False
    compile_gpu_chunk_eval: bool = False
    compile_gpu_chunk_size: int = 4
    cpu_worker_mode: str = "pytorch"
    cpu_worker_nice: int = 0
    gpu_slowdown_factor: float = 1.0
    output_root: str = "outputs"
    output_formats: List[str] = field(default_factory=lambda: ["csv"])


@dataclass
class ResourcesConfig:
    gpus: int = 1
    ranks: int = 1
    cpu_workers: int = 1


@dataclass
class DecompositionConfig:
    strategy: str = "row"
    tile_rows: int = 2
    tile_cols: int = 2
    chunk_size: int = 8
    cpu_schedule: str = "dynamic"
    cpu_chunk_size: int = 1
    gpu_chunk_size_max: int = 8
    fixed_gpu_chunk_size: Optional[int] = None
    calibration_points: int = 1
    cpu_threads_per_worker: int = 1


@dataclass
class MpiConfig:
    communication_mode: str = "gather"
    expected_world_size: int = 1
    overlap_chunk_size: int = 8


@dataclass
class ScalingReference:
    baseline_workers: int = 1
    baseline_time_s: Optional[float] = None


@dataclass
class ReferenceConfig:
    strong_scaling: ScalingReference = field(default_factory=ScalingReference)
    weak_scaling: ScalingReference = field(default_factory=ScalingReference)


@dataclass
class ExperimentConfig:
    experiment_name: str = "loss-grid"
    seed: int = 1337
    backend: str = "gpu"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    mpi: MpiConfig = field(default_factory=MpiConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    sweep: Dict[str, List[Any]] = field(default_factory=dict)
    cases: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def clone(self) -> "ExperimentConfig":
        return experiment_config_from_dict(copy.deepcopy(self.to_dict()))


def _load_raw(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            return json.load(handle)
        return yaml.safe_load(handle)


def _merge_dataclass(cls: Any, raw: Optional[Dict[str, Any]]) -> Any:
    raw = raw or {}
    return cls(**raw)


def experiment_config_from_dict(raw: Dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "loss-grid"),
        seed=raw.get("seed", 1337),
        backend=raw.get("backend", "gpu"),
        model=_merge_dataclass(ModelConfig, raw.get("model")),
        data=_merge_dataclass(DataConfig, raw.get("data")),
        grid=_merge_dataclass(GridConfig, raw.get("grid")),
        runtime=_merge_dataclass(RuntimeConfig, raw.get("runtime")),
        resources=_merge_dataclass(ResourcesConfig, raw.get("resources")),
        decomposition=_merge_dataclass(DecompositionConfig, raw.get("decomposition")),
        mpi=_merge_dataclass(MpiConfig, raw.get("mpi")),
        reference=ReferenceConfig(
            strong_scaling=_merge_dataclass(
                ScalingReference,
                (raw.get("reference") or {}).get("strong_scaling"),
            ),
            weak_scaling=_merge_dataclass(
                ScalingReference,
                (raw.get("reference") or {}).get("weak_scaling"),
            ),
        ),
        sweep=raw.get("sweep", {}) or {},
        cases=raw.get("cases", []) or [],
    )


def load_config(path: str) -> ExperimentConfig:
    return experiment_config_from_dict(_load_raw(path))
