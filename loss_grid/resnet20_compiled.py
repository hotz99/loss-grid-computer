from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

def build_parameter_vector(
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    return base_vector + (alpha * direction_a) + (beta * direction_b)


def split_parameter_vector(
    parameter_vector: torch.Tensor,
    parameter_numels: tuple[int, ...],
    parameter_shapes: tuple[torch.Size, ...],
) -> tuple[torch.Tensor, ...]:
    parameters = []
    offset = 0
    for numel, shape in zip(parameter_numels, parameter_shapes):
        next_offset = offset + numel
        parameters.append(parameter_vector[offset:next_offset].view(shape))
        offset = next_offset
    return tuple(parameters)


def build_resnet20_bn_stats(buffers: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    names = (
        "bn1",
        "layer1.0.bn1",
        "layer1.0.bn2",
        "layer1.1.bn1",
        "layer1.1.bn2",
        "layer1.2.bn1",
        "layer1.2.bn2",
        "layer2.0.bn1",
        "layer2.0.bn2",
        "layer2.0.downsample.1",
        "layer2.1.bn1",
        "layer2.1.bn2",
        "layer2.2.bn1",
        "layer2.2.bn2",
        "layer3.0.bn1",
        "layer3.0.bn2",
        "layer3.0.downsample.1",
        "layer3.1.bn1",
        "layer3.1.bn2",
        "layer3.2.bn1",
        "layer3.2.bn2",
    )
    stats = []
    for name in names:
        stats.append(buffers[f"{name}.running_mean"])
        stats.append(buffers[f"{name}.running_var"])
    return tuple(stats)


def _bn_eval(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
) -> torch.Tensor:
    return F.batch_norm(
        inputs,
        running_mean=running_mean,
        running_var=running_var,
        weight=weight,
        bias=bias,
        training=False,
        momentum=0.0,
        eps=1e-5,
    )


def _basic_block(
    inputs: torch.Tensor,
    params: tuple[torch.Tensor, ...],
    bn_stats: tuple[torch.Tensor, ...],
    conv1_idx: int,
    bn1_weight_idx: int,
    bn1_bias_idx: int,
    bn1_stats_idx: int,
    conv2_idx: int,
    bn2_weight_idx: int,
    bn2_bias_idx: int,
    bn2_stats_idx: int,
    stride: int,
    use_skip: bool,
    downsample_conv_idx: Optional[int] = None,
    downsample_bn_weight_idx: Optional[int] = None,
    downsample_bn_bias_idx: Optional[int] = None,
    downsample_bn_stats_idx: Optional[int] = None,
) -> torch.Tensor:
    residual = inputs
    outputs = F.conv2d(inputs, params[conv1_idx], bias=None, stride=stride, padding=1)
    outputs = _bn_eval(
        outputs,
        params[bn1_weight_idx],
        params[bn1_bias_idx],
        bn_stats[bn1_stats_idx],
        bn_stats[bn1_stats_idx + 1],
    )
    outputs = F.relu(outputs, inplace=False)
    outputs = F.conv2d(outputs, params[conv2_idx], bias=None, stride=1, padding=1)
    outputs = _bn_eval(
        outputs,
        params[bn2_weight_idx],
        params[bn2_bias_idx],
        bn_stats[bn2_stats_idx],
        bn_stats[bn2_stats_idx + 1],
    )

    if use_skip:
        if downsample_conv_idx is not None:
            residual = F.conv2d(
                inputs,
                params[downsample_conv_idx],
                bias=None,
                stride=stride,
                padding=0,
            )
            residual = _bn_eval(
                residual,
                params[downsample_bn_weight_idx],
                params[downsample_bn_bias_idx],
                bn_stats[downsample_bn_stats_idx],
                bn_stats[downsample_bn_stats_idx + 1],
            )
        outputs = outputs + residual

    return F.relu(outputs, inplace=False)


def resnet20_forward_from_params(
    inputs: torch.Tensor,
    params: tuple[torch.Tensor, ...],
    bn_stats: tuple[torch.Tensor, ...],
    use_skip: bool,
) -> torch.Tensor:
    outputs = F.conv2d(inputs, params[0], bias=None, stride=1, padding=1)
    outputs = _bn_eval(outputs, params[1], params[2], bn_stats[0], bn_stats[1])
    outputs = F.relu(outputs, inplace=False)

    outputs = _basic_block(outputs, params, bn_stats, 3, 4, 5, 2, 6, 7, 8, 4, 1, use_skip)
    outputs = _basic_block(outputs, params, bn_stats, 9, 10, 11, 6, 12, 13, 14, 8, 1, use_skip)
    outputs = _basic_block(outputs, params, bn_stats, 15, 16, 17, 10, 18, 19, 20, 12, 1, use_skip)

    outputs = _basic_block(
        outputs,
        params,
        bn_stats,
        21,
        22,
        23,
        14,
        24,
        25,
        26,
        16,
        2,
        use_skip,
        27,
        28,
        29,
        18,
    )
    outputs = _basic_block(outputs, params, bn_stats, 30, 31, 32, 20, 33, 34, 35, 22, 1, use_skip)
    outputs = _basic_block(outputs, params, bn_stats, 36, 37, 38, 24, 39, 40, 41, 26, 1, use_skip)

    outputs = _basic_block(
        outputs,
        params,
        bn_stats,
        42,
        43,
        44,
        28,
        45,
        46,
        47,
        30,
        2,
        use_skip,
        48,
        49,
        50,
        32,
    )
    outputs = _basic_block(outputs, params, bn_stats, 51, 52, 53, 34, 54, 55, 56, 36, 1, use_skip)
    outputs = _basic_block(outputs, params, bn_stats, 57, 58, 59, 38, 60, 61, 62, 40, 1, use_skip)

    outputs = F.adaptive_avg_pool2d(outputs, (1, 1))
    outputs = torch.flatten(outputs, 1)
    return F.linear(outputs, params[63], params[64])


def build_resnet20_compiled_chunk_evaluator(
    model_name: str,
    use_skip: bool,
    base_vector: torch.Tensor,
    direction_a: torch.Tensor,
    direction_b: torch.Tensor,
    parameter_numels: tuple[int, ...],
    parameter_shapes: tuple[torch.Size, ...],
    buffers: dict[str, torch.Tensor],
    example_inputs: torch.Tensor,
    example_targets: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]:
    if device.type == "cpu" or model_name.lower() not in {"resnet20", "resnet20_no_skip"}:
        return None

    bn_stats = build_resnet20_bn_stats(buffers)
    fixed_chunk_size = max(1, int(chunk_size))

    def eval_chunk(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        perturbations: torch.Tensor,
    ) -> torch.Tensor:
        losses = []
        for index in range(fixed_chunk_size):
            alpha = perturbations[index, 0]
            beta = perturbations[index, 1]
            parameter_vector = build_parameter_vector(
                base_vector,
                direction_a,
                direction_b,
                alpha,
                beta,
            )
            params = split_parameter_vector(
                parameter_vector=parameter_vector,
                parameter_numels=parameter_numels,
                parameter_shapes=parameter_shapes,
            )
            logits = resnet20_forward_from_params(
                inputs=inputs,
                params=params,
                bn_stats=bn_stats,
                use_skip=use_skip,
            )
            losses.append(F.cross_entropy(logits, targets))
        return torch.stack(losses)

    def _verify_candidate(
        candidate: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        reference: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[bool, str]:
        sample_perturbations = torch.zeros(
            (fixed_chunk_size, 2),
            device=device,
            dtype=base_vector.dtype,
        )
        if fixed_chunk_size >= 2:
            sample_perturbations[1, 0] = 0.5
            sample_perturbations[1, 1] = -0.25
        if fixed_chunk_size >= 3:
            sample_perturbations[2, 0] = 0.3
            sample_perturbations[2, 1] = 0.7
        if fixed_chunk_size >= 4:
            sample_perturbations[3, 0] = -0.4
            sample_perturbations[3, 1] = -0.6

        with torch.inference_mode():
            expected = reference(example_inputs, example_targets, sample_perturbations)
            actual = candidate(example_inputs, example_targets, sample_perturbations)

        if not torch.isfinite(actual).all():
            return False, "nonfinite_outputs"
        if not torch.allclose(expected, actual, atol=1e-4, rtol=1e-4):
            max_diff = float((expected - actual).abs().max().detach().cpu())
            return False, f"verification_mismatch max_diff={max_diff:.6e}"
        return True, "ok"

    try:
        compiled = torch.compile(eval_chunk, fullgraph=True, dynamic=False)
        verified, reason = _verify_candidate(compiled, eval_chunk)
        if not verified:
            print(
                f"[compile_chunk] disabled model={model_name} device={device.type} "
                f"reason={reason} fallback=eager_chunk"
            )
            return eval_chunk
        print(
            f"[compile_chunk] enabled model={model_name} device={device.type} "
            f"chunk_size={fixed_chunk_size} fullgraph=True"
        )
        return compiled
    except Exception as error:
        print(
            f"[compile_chunk] disabled model={model_name} device={device.type} "
            f"reason={error} fallback=eager_chunk"
        )
        return eval_chunk
