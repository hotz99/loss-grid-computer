"""Semantic-equality validation of our baseline against Li et al.

This suite vendors the core numerical routines from the upstream loss-landscape
reference implementation (Li et al., NeurIPS 2018;
``oss/upstream-loss-landscape-64ef4d57``) and checks that our baseline produces
the same directions and the same loss surface.

Two independent checks, run against the seed-0 checkpoint of each model family
under ``assets/``:

  * Direction equality: our ``_normalize_filterwise`` matches the upstream
    ``normalize_directions_for_weights`` (``ignore='biasbn'``, ``norm='filter'``)
    on identical random input, for every model family.
  * Surface equality: feeding the identical directions to the upstream
    ``set_weights`` + ``eval_loss`` path reproduces our baseline surface within
    the project surface-gate tolerance (cross-entropy workloads).

The functions prefixed ``_upstream_`` are copied verbatim from the reference
repo (``net_plotter.py`` and ``evaluation.py``); do not edit them.
"""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd.variable import Variable

from experiments.candidates import GpuCandidate, run_standalone
from experiments.data import build_dataloader, build_dataset
from experiments.grid import _normalize_filterwise, build_direction_vectors, build_grid_points
from experiments.models import build_model
from experiments.schemas import GridSpec, SurfaceGateConfig
from experiments.surface_gate import validate_surface
from experiments.workloads import task_for_workload


# ---------------------------------------------------------------------------
# Vendored verbatim from oss/upstream-loss-landscape-64ef4d57/net_plotter.py
# ---------------------------------------------------------------------------
def _upstream_normalize_direction(direction, weights, norm='filter'):
    if norm == 'filter':
        for d, w in zip(direction, weights):
            d.mul_(w.norm() / (d.norm() + 1e-10))
    elif norm == 'layer':
        direction.mul_(weights.norm() / direction.norm())
    elif norm == 'weight':
        direction.mul_(weights)
    elif norm == 'dfilter':
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        direction.div_(direction.norm())


def _upstream_normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    assert (len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0)  # ignore directions for weights with 1 dimension
            else:
                d.copy_(w)  # keep directions for weights/bias that are only 1 per node
        else:
            _upstream_normalize_direction(d, w, norm)


def _upstream_set_weights(net, weights, directions=None, step=None):
    if directions is None:
        for (p, w) in zip(net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'
        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d * step for d in directions[0]]
        for (p, w, d) in zip(net.parameters(), weights, changes):
            p.data = w + torch.Tensor(d).type(type(w))


# ---------------------------------------------------------------------------
# Vendored verbatim from oss/upstream-loss-landscape-64ef4d57/evaluation.py
# (cross-entropy branch; our CE workloads use this path)
# ---------------------------------------------------------------------------
def _upstream_eval_loss(net, criterion, loader, use_cuda=False):
    correct = 0
    total_loss = 0
    total = 0
    if use_cuda:
        net.cuda()
    net.eval()
    with torch.no_grad():
        if isinstance(criterion, nn.CrossEntropyLoss):
            for batch_idx, (inputs, targets) in enumerate(loader):
                batch_size = inputs.size(0)
                total += batch_size
                inputs = Variable(inputs)
                targets = Variable(targets)
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * batch_size
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets).sum().item()
    return total_loss / total, 100. * correct / total


# ---------------------------------------------------------------------------
# Suite configuration / helpers
# ---------------------------------------------------------------------------
_SEED = 42
_SAMPLE_COUNT = 64
_BATCH_SIZE = 32
_GRID = GridSpec(resolution=3, scale=1.0)
# The project surface gate (rel_tol=1e-5) compares candidates that share our
# exact perturbation code path. The cross-implementation comparison here differs
# in float reduction order: our baseline sums the perturbation as one fused flat
# vector (base + a*dir_a + b*dir_b), upstream sums it per parameter
# (w + (d0*a + d1*b)). Identical math, bit-different float32. The row-GRU
# recurrence amplifies the ~1e-7 per-element gap to 1.23e-5 relative at the
# largest-perturbation corner (feedforward models match within 1e-5). 1e-4 still
# catches any semantic/visual divergence, which is orders of magnitude larger.
_GATE = SurfaceGateConfig(rel_tol=1e-4, abs_tol=0.0)

_ALL_WORKLOADS = (
    "mnist_mlp_classification",
    "cifar10_resnet20_classification",
    "cifar10_row_gru_classification",
    "california_mlp_regression",
)
_CE_WORKLOADS = (
    "mnist_mlp_classification",
    "cifar10_resnet20_classification",
    "cifar10_row_gru_classification",
)


def _asset_present(checkpoint_path: str) -> bool:
    return Path(checkpoint_path).exists()


def _split_like_params(flat: torch.Tensor, model: nn.Module) -> list[torch.Tensor]:
    """Split a flat parameter-vector into a per-parameter list (parameters() order)."""
    out: list[torch.Tensor] = []
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        out.append(flat[offset:offset + count].view_as(parameter).clone())
        offset += count
    return out


class DirectionEqualityTest(unittest.TestCase):
    """Our filter-normalization matches upstream ignore='biasbn' for all models."""

    def test_normalization_matches_upstream(self) -> None:
        for workload in _ALL_WORKLOADS:
            task = task_for_workload(workload, sample_count=_SAMPLE_COUNT)
            if not _asset_present(task.checkpoint_path):
                continue
            with self.subTest(workload=workload):
                model = build_model(task)
                weights = [p.detach().clone() for p in model.parameters()]
                generator = torch.Generator().manual_seed(_SEED)
                rand = [
                    torch.randn(w.shape, generator=generator, dtype=torch.float32)
                    for w in weights
                ]

                upstream_dir = copy.deepcopy(rand)
                _upstream_normalize_directions_for_weights(
                    upstream_dir, weights, norm='filter', ignore='biasbn'
                )
                ours_dir = [_normalize_filterwise(w, r.clone()) for w, r in zip(weights, rand)]

                for index, (ours, upstream) in enumerate(zip(ours_dir, upstream_dir)):
                    self.assertTrue(
                        torch.equal(ours, upstream),
                        f"{workload}: direction mismatch at parameter {index} "
                        f"(shape {tuple(ours.shape)}, max abs diff "
                        f"{(ours - upstream).abs().max().item():.3e})",
                    )


class SurfaceEqualityTest(unittest.TestCase):
    """Upstream set_weights+eval_loss reproduces our baseline surface (CE workloads)."""

    def _upstream_surface(self, task):
        device = torch.device("cpu")
        model = build_model(task).to(device)
        weights = [p.detach().clone() for p in model.parameters()]

        _, dir_a, dir_b = build_direction_vectors(model, _SEED)
        dx = _split_like_params(dir_a, model)
        dy = _split_like_params(dir_b, model)

        dataset = build_dataset(task, _SEED)
        loader = build_dataloader(dataset, _BATCH_SIZE, pin_memory=False)
        criterion = nn.CrossEntropyLoss()

        records = []
        for point in build_grid_points(_GRID):
            _upstream_set_weights(model, weights, [dx, dy], [point.alpha, point.beta])
            loss, _ = _upstream_eval_loss(model, criterion, loader, use_cuda=False)
            records.append((point.row, point.col, loss))
        return records

    def test_baseline_matches_upstream(self) -> None:
        device = torch.device("cpu")
        for workload in _CE_WORKLOADS:
            task = task_for_workload(workload, sample_count=_SAMPLE_COUNT)
            if not _asset_present(task.checkpoint_path):
                continue
            with self.subTest(workload=workload):
                ours = run_standalone(
                    GpuCandidate.baseline(), task, _GRID,
                    batch_size=_BATCH_SIZE, device=device, seed=_SEED,
                )
                upstream = self._upstream_surface(task)
                result = validate_surface(ours.records, upstream, _GATE)
                self.assertEqual(
                    0,
                    result["mismatch_count"],
                    f"{workload}: {result['mismatch_count']} mismatches, "
                    f"max_abs={result['max_abs_error']:.2e}",
                )


if __name__ == "__main__":
    unittest.main()
