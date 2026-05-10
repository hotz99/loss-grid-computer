from __future__ import annotations

import unittest

import torch
from torch.func import functional_call, vmap
from torch.nn.utils import parameters_to_vector

from src.functional_eval.layout import (
    build_parameter_layout,
    extract_named_buffers,
    extract_named_parameters,
    flat_chunk_to_batched_param_dict,
    flat_vector_to_param_dict,
    make_functional_state,
)


class FunctionalEvalLayoutTest(unittest.TestCase):
    def test_extracts_named_parameters_in_module_order(self) -> None:
        model = _SmallModule()

        named_parameters = extract_named_parameters(model)

        self.assertEqual(
            ["linear.weight", "linear.bias", "head.weight"],
            list(named_parameters.keys()),
        )
        for parameter in named_parameters.values():
            self.assertFalse(parameter.requires_grad)

    def test_extracts_named_buffers_in_module_order(self) -> None:
        model = _SmallModule()

        named_buffers = extract_named_buffers(model)

        self.assertEqual(
            [
                "scale",
                "batch_norm.running_mean",
                "batch_norm.running_var",
                "batch_norm.num_batches_tracked",
            ],
            list(named_buffers.keys()),
        )
        for buffer in named_buffers.values():
            self.assertFalse(buffer.requires_grad)

    def test_builds_offsets_shapes_and_total_length(self) -> None:
        model = _SmallModule()
        named_parameters = extract_named_parameters(model)

        layout = build_parameter_layout(named_parameters.items())

        self.assertEqual(13, layout.total_numel)
        self.assertEqual(
            [
                ("linear.weight", 0, 6, (2, 3)),
                ("linear.bias", 6, 2, (2,)),
                ("head.weight", 8, 5, (1, 5)),
            ],
            [
                (entry.name, entry.offset, entry.numel, entry.shape)
                for entry in layout.entries
            ],
        )

    def test_reconstructs_parameter_dict_from_flat_vector(self) -> None:
        model = _SmallModule()
        named_parameters, _named_buffers, layout = make_functional_state(model)
        flat_vector = parameters_to_vector(named_parameters.values())

        reconstructed = flat_vector_to_param_dict(flat_vector, layout)

        self.assertEqual(list(named_parameters.keys()), list(reconstructed.keys()))
        for name, parameter in named_parameters.items():
            torch.testing.assert_close(reconstructed[name], parameter)

    def test_reconstructed_parameters_are_views_into_flat_vector(self) -> None:
        model = _SmallModule()
        named_parameters, _named_buffers, layout = make_functional_state(model)
        flat_vector = parameters_to_vector(named_parameters.values()).clone()

        reconstructed = flat_vector_to_param_dict(flat_vector, layout)
        flat_vector[0] = 123.0

        self.assertEqual(torch.tensor(123.0), reconstructed["linear.weight"][0, 0])

    def test_rejects_flat_vector_with_wrong_shape_or_length(self) -> None:
        model = _SmallModule()
        named_parameters, _named_buffers, layout = make_functional_state(model)
        flat_vector = parameters_to_vector(named_parameters.values())

        with self.assertRaisesRegex(ValueError, "expected a 1D"):
            flat_vector_to_param_dict(flat_vector.unsqueeze(0), layout)
        with self.assertRaisesRegex(ValueError, "does not match layout"):
            flat_vector_to_param_dict(flat_vector[:-1], layout)

    def test_builds_batched_parameter_dict_from_flat_chunk(self) -> None:
        model = _SmallModule()
        named_parameters, _named_buffers, layout = make_functional_state(model)
        base = parameters_to_vector(named_parameters.values())
        flat_vectors = torch.stack([base, base + 1.0, base - 2.0])

        batched = flat_chunk_to_batched_param_dict(flat_vectors, layout)

        self.assertEqual(list(named_parameters.keys()), list(batched.keys()))
        self.assertEqual((3, 2, 3), tuple(batched["linear.weight"].shape))
        self.assertEqual((3, 2), tuple(batched["linear.bias"].shape))
        self.assertEqual((3, 1, 5), tuple(batched["head.weight"].shape))
        torch.testing.assert_close(
            batched["linear.weight"][1],
            named_parameters["linear.weight"] + 1.0,
        )

    def test_batched_parameters_are_views_into_flat_chunk(self) -> None:
        model = _SmallModule()
        named_parameters, _named_buffers, layout = make_functional_state(model)
        base = parameters_to_vector(named_parameters.values())
        flat_vectors = torch.stack([base, base + 1.0])

        batched = flat_chunk_to_batched_param_dict(flat_vectors, layout)
        flat_vectors[1, 0] = 321.0

        torch.testing.assert_close(
            batched["linear.weight"][1, 0, 0],
            torch.tensor(321.0),
        )

    def test_batched_parameter_dict_can_drive_vmap_functional_call(self) -> None:
        model = _SmallModule().eval()
        named_parameters, named_buffers, layout = make_functional_state(model)
        base = parameters_to_vector(named_parameters.values())
        flat_vectors = torch.stack([base, base + 0.5])
        batched_parameters = flat_chunk_to_batched_param_dict(flat_vectors, layout)
        inputs = torch.randn(4, 3)

        def call_with_params(params: dict[str, torch.Tensor]) -> torch.Tensor:
            return functional_call(model, (params, named_buffers), (inputs,))

        outputs = vmap(call_with_params)(batched_parameters)

        self.assertEqual((2, 4, 1), tuple(outputs.shape))
        torch.testing.assert_close(
            outputs[0],
            functional_call(
                model,
                (flat_vector_to_param_dict(base, layout), named_buffers),
                (inputs,),
            ),
        )

    def test_rejects_flat_chunk_with_wrong_shape_or_length(self) -> None:
        model = _SmallModule()
        named_parameters, _named_buffers, layout = make_functional_state(model)
        flat_vector = parameters_to_vector(named_parameters.values())

        with self.assertRaisesRegex(ValueError, "expected a 2D"):
            flat_chunk_to_batched_param_dict(flat_vector, layout)
        with self.assertRaisesRegex(ValueError, "does not match layout"):
            flat_chunk_to_batched_param_dict(torch.stack([flat_vector[:-1]]), layout)


class _SmallModule(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(3, 2)
        self.register_buffer("scale", torch.tensor([2.0]))
        self.batch_norm = torch.nn.BatchNorm1d(2, affine=False)
        self.head = torch.nn.Linear(5, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.batch_norm(self.linear(inputs))
        scaled = features * self.scale
        padded = torch.cat([scaled, inputs], dim=1)
        return self.head(padded)


if __name__ == "__main__":
    unittest.main()
