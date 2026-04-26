from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
from importlib import import_module
import json
from pathlib import Path
from typing import Any, Literal, Optional


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
    scale: float = 1.0


@dataclass
class RuntimeConfig:
    device: Literal["auto", "mps", "cuda", "cpu"] = "auto"
    num_batches: Optional[int] = 4
    preload: bool = False
    gpu_slowdown_factor: float = 1.0
    validation_baseline_config: Optional[str] = None
    output_root: str = "outputs"
    verbose: bool = False


@dataclass
class ResourcesConfig:
    cpu_workers: int = 1


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

    def clone(self) -> "ExperimentConfig":
        return copy.deepcopy(self)


@dataclass(frozen=True)
class VanillaExecutionConfig:
    workload: ExperimentConfig
    _tag: Literal["vanilla"] = "vanilla"


@dataclass(frozen=True)
class HybridExecutionConfig:
    workload: ExperimentConfig
    cpu_workers: int
    cpu_batch_size: int
    _tag: Literal["hybrid"] = "hybrid"


def experiment_config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)


def _load_raw(path: str) -> dict[str, Any]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() == ".json":
            return json.load(handle)
        try:
            yaml = import_module("yaml")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyYAML is required to load YAML configs. "
                "Install dependencies with `pip install -r requirements.txt`."
            ) from exc
        return yaml.safe_load(handle)


def experiment_config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    runtime_raw = dict(raw.get("runtime") or {})
    if "preload" not in runtime_raw:
        if "preload_device_batches" in runtime_raw:
            runtime_raw["preload"] = runtime_raw["preload_device_batches"]
        elif "preload_gpu_batches" in runtime_raw:
            runtime_raw["preload"] = runtime_raw["preload_gpu_batches"]

    return ExperimentConfig(
        experiment_name=raw.get("experiment_name", "loss-grid"),
        seed=raw.get("seed", 1337),
        backend=raw.get("backend", "vanilla"),
        model=ModelConfig(**(raw.get("model") or {})),
        data=DataConfig(**(raw.get("data") or {})),
        grid=GridConfig(**(raw.get("grid") or {})),
        runtime=RuntimeConfig(**runtime_raw),
        resources=ResourcesConfig(**(raw.get("resources") or {})),
    )


def validate_config(config: ExperimentConfig) -> None:
    total_points = config.grid.resolution * config.grid.resolution
    if config.backend not in {"vanilla", "hybrid"}:
        raise ValueError(f"Unsupported backend: {config.backend}")
    if config.runtime.device not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError("runtime.device must be one of: auto, cuda, mps, cpu")
    if config.grid.resolution < 1:
        raise ValueError("grid.resolution must be >= 1")
    if config.grid.scale <= 0:
        raise ValueError("grid.scale must be > 0")
    if config.data.batch_size < 1:
        raise ValueError("data.batch_size must be >= 1")
    if config.data.cpu_batch_size is not None and config.data.cpu_batch_size < 1:
        raise ValueError("data.cpu_batch_size must be >= 1 when set")
    if config.data.gpu_batch_size is not None and config.data.gpu_batch_size < 1:
        raise ValueError("data.gpu_batch_size must be >= 1 when set")
    if config.resources.cpu_workers < 0:
        raise ValueError("resources.cpu_workers must be >= 0")
    if config.backend == "hybrid" and total_points < 2:
        raise ValueError("hybrid requires at least 2 grid points")
    if config.runtime.num_batches is not None and config.runtime.num_batches < 1:
        raise ValueError("runtime.num_batches must be >= 1 when set")
    if config.runtime.gpu_slowdown_factor < 1.0:
        raise ValueError("runtime.gpu_slowdown_factor must be >= 1.0")


def validate_rq1_config(
    repeats: int,
    max_slowdown: float,
    jump_factor: float,
    linear_samples: int,
) -> None:
    if repeats < 1:
        raise ValueError("--repeats must be >= 1")
    if max_slowdown < 1.0:
        raise ValueError("--max-slowdown must be >= 1.0")
    if jump_factor <= 1.0:
        raise ValueError("--jump-factor must be > 1.0")
    if linear_samples < 2:
        raise ValueError("--linear-samples must be >= 2")


def load_config(path: str) -> ExperimentConfig:
    config = experiment_config_from_dict(_load_raw(path))
    validate_config(config)
    return config
