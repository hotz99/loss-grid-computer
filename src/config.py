from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ModelConfig:
    checkpoint_path: Optional[str] = None


@dataclass
class DataConfig:
    root: str = "assets/cifar-10-batches-py"
    subset_size: int = 256
    batch_size: int = 32
    cpu_batch_size: Optional[int] = None
    gpu_batch_size: Optional[int] = None
    num_workers: int = 0


@dataclass
class GridConfig:
    resolution: int = 9


@dataclass
class RuntimeConfig:
    num_batches: Optional[int] = 4
    preload_gpu_batches: bool = False
    preload_max_batches: Optional[int] = None
    compile_gpu_chunk_eval: bool = False
    compile_gpu_chunk_size: Optional[int] = 4
    gpu_slowdown_factor: float = 1.0
    output_root: str = "outputs"


@dataclass
class ResourcesConfig:
    cpu_workers: int = 1


@dataclass
class DecompositionConfig:
    cpu_chunk_size: int = 1
    gpu_chunk_size_max: int = 8
    fixed_gpu_chunk_size: Optional[int] = None
    gpu_initial_ratio: float = 0.5
    cpu_threads_per_worker: int = 1


@dataclass
class ExperimentConfig:
    experiment_name: str = "loss-grid"
    seed: int = 1337
    backend: str = "vanilla"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    resources: ResourcesConfig = field(default_factory=ResourcesConfig)
    decomposition: DecompositionConfig = field(default_factory=DecompositionConfig)
    sweep: Dict[str, List[Any]] = field(default_factory=dict)

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


def experiment_config_from_dict(raw: Dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "loss-grid"),
        seed=raw.get("seed", 1337),
        backend=raw.get("backend", "gpu"),
        model=ModelConfig(**(raw.get("model") or {})),
        data=DataConfig(**(raw.get("data") or {})),
        grid=GridConfig(**(raw.get("grid") or {})),
        runtime=RuntimeConfig(**(raw.get("runtime") or {})),
        resources=ResourcesConfig(**(raw.get("resources") or {})),
        decomposition=DecompositionConfig(**(raw.get("decomposition") or {})),
        sweep=raw.get("sweep", {}) or {},
    )


def _validate_sweep(config: ExperimentConfig) -> None:
    for key, values in config.sweep.items():
        if not isinstance(values, list) or not values:
            raise ValueError(f"sweep.{key} must be a non-empty list")
        if key == "resources.cpu_workers":
            for value in values:
                if value < 1:
                    raise ValueError("sweep.resources.cpu_workers values must be >= 1")
        elif key == "data.cpu_batch_size":
            for value in values:
                if value < 1:
                    raise ValueError("sweep.data.cpu_batch_size values must be >= 1")
        elif key == "decomposition.cpu_chunk_size":
            for value in values:
                if value < 1:
                    raise ValueError(
                        "sweep.decomposition.cpu_chunk_size values must be >= 1"
                    )
        elif key == "decomposition.gpu_initial_ratio":
            for value in values:
                if not 0.0 <= value <= 1.0:
                    raise ValueError(
                        "sweep.decomposition.gpu_initial_ratio values must be in [0, 1]"
                    )
        else:
            raise ValueError(f"Unsupported sweep key: {key}")


def validate_config(config: ExperimentConfig) -> None:
    if config.backend not in {"vanilla", "vanilla_compiled", "hybrid"}:
        raise ValueError(f"Unsupported backend: {config.backend}")
    if config.grid.resolution < 1:
        raise ValueError("grid.resolution must be >= 1")
    if config.data.batch_size < 1:
        raise ValueError("data.batch_size must be >= 1")
    if config.data.cpu_batch_size is not None and config.data.cpu_batch_size < 1:
        raise ValueError("data.cpu_batch_size must be >= 1 when set")
    if config.data.gpu_batch_size is not None and config.data.gpu_batch_size < 1:
        raise ValueError("data.gpu_batch_size must be >= 1 when set")
    if config.resources.cpu_workers < 0:
        raise ValueError("resources.cpu_workers must be >= 0")
    if config.backend == "hybrid" and config.resources.cpu_workers < 1:
        raise ValueError("resources.cpu_workers must be >= 1 for hybrid")
    if config.decomposition.cpu_chunk_size < 1:
        raise ValueError("decomposition.cpu_chunk_size must be >= 1")
    if config.decomposition.gpu_chunk_size_max < 1:
        raise ValueError("decomposition.gpu_chunk_size_max must be >= 1")
    if (
        config.decomposition.fixed_gpu_chunk_size is not None
        and config.decomposition.fixed_gpu_chunk_size < 1
    ):
        raise ValueError("decomposition.fixed_gpu_chunk_size must be >= 1 when set")
    if not 0.0 <= config.decomposition.gpu_initial_ratio <= 1.0:
        raise ValueError("decomposition.gpu_initial_ratio must be in [0, 1]")
    if config.decomposition.cpu_threads_per_worker < 1:
        raise ValueError("decomposition.cpu_threads_per_worker must be >= 1")
    if config.runtime.num_batches is not None and config.runtime.num_batches < 1:
        raise ValueError("runtime.num_batches must be >= 1 when set")
    if (
        config.runtime.preload_max_batches is not None
        and config.runtime.preload_max_batches < 1
    ):
        raise ValueError("runtime.preload_max_batches must be >= 1 when set")
    if (
        config.runtime.compile_gpu_chunk_size is not None
        and config.runtime.compile_gpu_chunk_size < 1
    ):
        raise ValueError("runtime.compile_gpu_chunk_size must be >= 1 when set")
    if config.runtime.gpu_slowdown_factor < 1.0:
        raise ValueError("runtime.gpu_slowdown_factor must be >= 1.0")
    _validate_sweep(config)


def load_config(path: str) -> ExperimentConfig:
    config = experiment_config_from_dict(_load_raw(path))
    validate_config(config)
    return config
