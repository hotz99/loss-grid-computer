from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from experiments.schemas import MLTaskSpec


Batch = tuple[torch.Tensor, torch.Tensor]


def compute_loss(
    model: nn.Module,
    batch: Batch,
    device: torch.device,
    task: MLTaskSpec,
) -> tuple[torch.Tensor, int]:
    inputs, targets = batch
    if task.loss == "cross_entropy":
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = F.cross_entropy(logits, targets, reduction="mean")
        return loss, int(targets.shape[0])
    if task.loss == "mse":
        inputs = inputs.to(device, dtype=torch.float32, non_blocking=True)
        targets = targets.to(device, dtype=torch.float32, non_blocking=True)
        predictions = model(inputs).squeeze(-1)
        loss = F.mse_loss(predictions, targets, reduction="mean")
        return loss, int(targets.shape[0])
    raise ValueError(f"unsupported loss: {task.loss}")


def functional_loss(
    logits_or_predictions: torch.Tensor,
    targets: torch.Tensor,
    task: MLTaskSpec,
) -> torch.Tensor:
    """Used by vmapped/compiled candidates that compute logits via functional_call."""
    if task.loss == "cross_entropy":
        return F.cross_entropy(logits_or_predictions, targets, reduction="mean")
    if task.loss == "mse":
        predictions = logits_or_predictions.squeeze(-1)
        targets = targets.to(dtype=predictions.dtype)
        diff = predictions - targets
        return (diff * diff).mean()
    raise ValueError(f"unsupported loss: {task.loss}")
