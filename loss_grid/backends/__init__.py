from loss_grid.backends.base import LossGridExecutor
from loss_grid.backends.gpu import GpuLossGridExecutor
from loss_grid.backends.hybrid import HybridLossGridExecutor
from loss_grid.backends.vanilla import VanillaGpuLossGridExecutor
from loss_grid.config import ExperimentConfig

try:
    from loss_grid.backends.mpi import MpiLossGridExecutor
    _HAS_MPI = True
except ImportError:
    _HAS_MPI = False


def make_executor(config: ExperimentConfig) -> LossGridExecutor:
    backend = config.backend.lower()
    if backend == "gpu":
        return GpuLossGridExecutor()
    if backend == "vanilla":
        return VanillaGpuLossGridExecutor()
    if backend == "hybrid":
        return HybridLossGridExecutor()
    if backend == "mpi":
        if not _HAS_MPI:
            raise ValueError("MPI backend not available")
        return MpiLossGridExecutor()
    raise ValueError(f"Unsupported backend: {config.backend}")


__all__ = ["LossGridExecutor", "make_executor", "VanillaGpuLossGridExecutor"]
