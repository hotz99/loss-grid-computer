from __future__ import annotations

from loss_grid.backends.gpu import GpuLossGridExecutor


class CompiledVanillaGpuLossGridExecutor(GpuLossGridExecutor):
    """Single-process GPU-only control using the compiled chunk evaluator."""

    pass
