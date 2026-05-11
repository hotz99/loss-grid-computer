from __future__ import annotations

import unittest

import torch
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import DataLoader, TensorDataset

from src.backends.base import GridPoint
from src.functional_eval.validation import compare_surfaces
from src.functional_eval.vmapped import evaluate_vmapped_points
from src.models.row_gru import RowGRUClassifier


class FunctionalEvalVmappedTest(unittest.TestCase):
    def test_cross_entropy_surface_matches_sequential_oracle(self) -> None:
        torch.manual_seed(7)
        device = torch.device("cpu")
        model = _Classifier().to(device).eval()
        loader = DataLoader(
            TensorDataset(
                torch.randn(5, 3),
                torch.tensor([0, 1, 2, 1, 0], dtype=torch.long),
            ),
            batch_size=2,
            shuffle=False,
        )
        points = _points()
        base = parameters_to_vector(model.parameters()).detach().clone()
        direction_a = torch.linspace(-0.05, 0.05, base.numel())
        direction_b = torch.linspace(0.03, -0.03, base.numel())

        result = evaluate_vmapped_points(
            model=model,
            data_loader=loader,
            device=device,
            points=points,
            base_vector=base,
            direction_a=direction_a,
            direction_b=direction_b,
            loss_name="cross_entropy",
            point_chunk_size=3,
        )

        expected = _sequential_surface(
            model_factory=_Classifier,
            original_state=model.state_dict(),
            loader=loader,
            points=points,
            base=base,
            direction_a=direction_a,
            direction_b=direction_b,
            loss_name="cross_entropy",
        )
        comparison = compare_surfaces(result.records, expected)
        self.assertTrue(comparison.allclose, comparison)
        self.assertTrue(result.succeeded)
        self.assertEqual("vmapped_functional_chunk_3", result.candidate)
        self.assertEqual(len(points), len(result.records))
        self.assertGreaterEqual(result.timings.total_grid_s, result.timings.batch_eval_s)

    def test_mse_surface_preserves_weighted_average_for_each_point(self) -> None:
        torch.manual_seed(11)
        device = torch.device("cpu")
        model = _Regressor().to(device).eval()
        loader = DataLoader(
            TensorDataset(
                torch.randn(7, 2),
                torch.linspace(-1.0, 1.0, 7),
            ),
            batch_size=3,
            shuffle=False,
        )
        points = _points()
        base = parameters_to_vector(model.parameters()).detach().clone()
        direction_a = torch.full_like(base, 0.02)
        direction_b = torch.linspace(-0.04, 0.04, base.numel())

        result = evaluate_vmapped_points(
            model=model,
            data_loader=loader,
            device=device,
            points=points,
            base_vector=base,
            direction_a=direction_a,
            direction_b=direction_b,
            loss_name="mse",
            point_chunk_size=2,
        )

        expected = _sequential_surface(
            model_factory=_Regressor,
            original_state=model.state_dict(),
            loader=loader,
            points=points,
            base=base,
            direction_a=direction_a,
            direction_b=direction_b,
            loss_name="mse",
        )
        comparison = compare_surfaces(result.records, expected)
        self.assertTrue(comparison.allclose, comparison)

    def test_mse_surface_handles_point_batched_predictions(self) -> None:
        torch.manual_seed(13)
        device = torch.device("cpu")
        model = _Regressor().to(device).eval()
        loader = DataLoader(
            TensorDataset(
                torch.randn(4, 2),
                torch.linspace(-0.5, 0.5, 4),
            ),
            batch_size=4,
            shuffle=False,
        )
        base = parameters_to_vector(model.parameters()).detach().clone()

        result = evaluate_vmapped_points(
            model=model,
            data_loader=loader,
            device=device,
            points=_points(),
            base_vector=base,
            direction_a=torch.full_like(base, 0.01),
            direction_b=torch.full_like(base, -0.02),
            loss_name="mse",
            point_chunk_size=4,
        )

        self.assertTrue(result.succeeded, result.error)
        self.assertEqual(4, len(result.records))

    def test_row_gru_surface_uses_vmap_compatible_forward(self) -> None:
        torch.manual_seed(17)
        device = torch.device("cpu")
        model = RowGRUClassifier(hidden_size=8).to(device).eval()
        loader = DataLoader(
            TensorDataset(
                torch.randn(3, 3, 32, 32),
                torch.tensor([0, 1, 2], dtype=torch.long),
            ),
            batch_size=2,
            shuffle=False,
        )
        points = _points()[:2]
        base = parameters_to_vector(model.parameters()).detach().clone()

        result = evaluate_vmapped_points(
            model=model,
            data_loader=loader,
            device=device,
            points=points,
            base_vector=base,
            direction_a=torch.full_like(base, 0.001),
            direction_b=torch.full_like(base, -0.001),
            loss_name="cross_entropy",
            point_chunk_size=2,
        )

        self.assertTrue(result.succeeded, result.error)
        self.assertEqual(len(points), len(result.records))

    def test_rejects_invalid_point_chunk_size(self) -> None:
        model = _Classifier().eval()
        base = parameters_to_vector(model.parameters()).detach().clone()

        with self.assertRaisesRegex(ValueError, "point_chunk_size"):
            evaluate_vmapped_points(
                model=model,
                data_loader=[],
                device=torch.device("cpu"),
                points=[],
                base_vector=base,
                direction_a=base,
                direction_b=base,
                loss_name="cross_entropy",
                point_chunk_size=0,
            )

    def test_oom_runtime_error_is_reported_as_failed_candidate(self) -> None:
        device = torch.device("cpu")
        model = _OomClassifier().to(device).eval()
        loader = DataLoader(
            TensorDataset(torch.randn(2, 3), torch.tensor([0, 1], dtype=torch.long)),
            batch_size=2,
        )
        base = parameters_to_vector(model.parameters()).detach().clone()

        result = evaluate_vmapped_points(
            model=model,
            data_loader=loader,
            device=device,
            points=_points()[:1],
            base_vector=base,
            direction_a=torch.zeros_like(base),
            direction_b=torch.zeros_like(base),
            loss_name="cross_entropy",
            point_chunk_size=1,
        )

        self.assertFalse(result.succeeded)
        self.assertEqual([], result.records)
        self.assertEqual("oom", result.metadata["failure_kind"])
        self.assertIn("out of memory", result.error or "")


def _points() -> list[GridPoint]:
    return [
        GridPoint(0, 0, 0, -0.5, -0.25),
        GridPoint(1, 0, 1, -0.5, 0.25),
        GridPoint(2, 1, 0, 0.5, -0.25),
        GridPoint(3, 1, 1, 0.5, 0.25),
    ]


def _sequential_surface(
    *,
    model_factory: type[torch.nn.Module],
    original_state: dict[str, torch.Tensor],
    loader: DataLoader,
    points: list[GridPoint],
    base: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    loss_name: str,
) -> list[tuple[int, int, float]]:
    model = model_factory().eval()
    model.load_state_dict(original_state)
    records: list[tuple[int, int, float]] = []

    for point in points:
        vector = base + point.alpha * direction_a + point.beta * direction_b
        vector_to_parameters(vector, model.parameters())
        total_loss = 0.0
        total_examples = 0
        with torch.no_grad():
            for inputs, targets in loader:
                outputs = model(inputs.float())
                if loss_name == "cross_entropy":
                    loss = torch.nn.functional.cross_entropy(outputs, targets)
                elif loss_name == "mse":
                    loss = torch.nn.functional.mse_loss(outputs.squeeze(-1), targets)
                else:  # pragma: no cover - test helper defensive branch
                    raise AssertionError(loss_name)
                batch_size = int(targets.shape[0])
                total_loss += float(loss) * batch_size
                total_examples += batch_size
        records.append((point.row, point.col, total_loss / max(1, total_examples)))
    return records


class _Classifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 5),
            torch.nn.Tanh(),
            torch.nn.Linear(5, 3),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class _Regressor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(2, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class _OomClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("CUDA out of memory while testing candidate failure")


if __name__ == "__main__":
    unittest.main()
