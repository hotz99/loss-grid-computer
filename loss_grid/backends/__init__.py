from loss_grid.backends.base import LossGridExecutor
from loss_grid.backends.gpu import GpuLossGridExecutor
from loss_grid.backends.hybrid import HybridLossGridExecutor
from loss_grid.backends.mpi import MpiLossGridExecutor
from loss_grid.config import ExperimentConfig


def make_executor(config: ExperimentConfig) -> LossGridExecutor:
    backend = config.backend.lower()
    if backend == "gpu":
        return GpuLossGridExecutor()
    if backend == "hybrid":
        return HybridLossGridExecutor()
    if backend == "mpi":
        return MpiLossGridExecutor()
    raise ValueError(f"Unsupported backend: {config.backend}")


__all__ = ["LossGridExecutor", "make_executor"]
