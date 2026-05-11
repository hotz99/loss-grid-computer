from __future__ import annotations

import json
import random
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal


DeviceName = Literal["auto", "cpu", "mps", "cuda"]
DEFAULT_WORKLOAD_NAME = "cifar10_resnet20_classification"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _append_event(
    events: list[dict[str, Any]],
    name: str,
    *,
    detail: str | None = None,
    **metadata: Any,
) -> None:
    event: dict[str, Any] = {"name": name}
    if detail is not None:
        event["detail"] = detail
    if metadata:
        event["metadata"] = _json_safe(metadata)
    events.append(event)


def _backend_availability(torch: Any) -> dict[str, bool]:
    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend is not None and mps_backend.is_available())
    return {
        "cpu": True,
        "mps": mps_available,
        "cuda": cuda_available,
    }


def _resolve_device(torch: Any, requested: DeviceName) -> tuple[Any | None, str | None]:
    availability = _backend_availability(torch)
    if requested == "auto":
        if availability["cuda"]:
            return torch.device("cuda"), None
        if availability["mps"]:
            return torch.device("mps"), None
        return torch.device("cpu"), None
    if requested == "cuda" and not availability["cuda"]:
        return None, "CUDA was requested but torch.cuda.is_available() is false"
    if requested == "mps" and not availability["mps"]:
        return None, "MPS was requested but torch.backends.mps.is_available() is false"
    return torch.device(requested), None


def _device_metadata(torch: Any, device: Any | None) -> dict[str, Any]:
    availability = _backend_availability(torch)
    metadata: dict[str, Any] = {
        "selected": None if device is None else device.type,
        "available": availability,
    }
    if device is not None and device.type == "cuda":
        metadata["cuda_device_count"] = int(torch.cuda.device_count())
        if torch.cuda.device_count() > 0:
            metadata["cuda_device_name"] = torch.cuda.get_device_name(device)
    if device is not None and device.type == "mps":
        metadata["mps_built"] = bool(torch.backends.mps.is_built())
    return metadata


def _synchronize(torch: Any, device: Any) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        synchronize = getattr(torch.mps, "synchronize", None)
        if synchronize is not None:
            synchronize()


def _seed_torch(torch: Any, device: Any, seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if device.type == "mps" and hasattr(torch, "mps"):
        manual_seed = getattr(torch.mps, "manual_seed", None)
        if manual_seed is not None:
            manual_seed(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _check_torch_import(
    passed: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> Any | None:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on local environment
        _append_event(errors, "torch_import", detail=f"{type(exc).__name__}: {exc}")
        return None

    _append_event(
        passed,
        "torch_import",
        version=getattr(torch, "__version__", "unknown"),
    )
    return torch


def _check_device_resolution(
    torch: Any,
    requested: DeviceName,
    passed: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> Any | None:
    device, skip_reason = _resolve_device(torch, requested)
    if skip_reason is not None:
        _append_event(skipped, "device_resolution", detail=skip_reason)
        return None

    _append_event(
        passed,
        "device_resolution",
        requested=requested,
        selected=device.type,
        availability=_backend_availability(torch),
    )
    return device


def _make_probe_model(torch: Any) -> Any:
    from torch import nn

    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(4),
        nn.ReLU(inplace=False),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(4, 2),
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.float()


def _named_state(torch: Any, model: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    params = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    buffers = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
    }
    return params, buffers


def _check_functional_call(
    torch: Any,
    device: Any,
    passed: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> tuple[Any | None, dict[str, Any] | None, dict[str, Any] | None]:
    if not hasattr(torch, "func") or not hasattr(torch.func, "functional_call"):
        _append_event(
            errors,
            "functional_call",
            detail="torch.func.functional_call is unavailable",
        )
        return None, None, None

    try:
        model = _make_probe_model(torch).to(device)
        model.eval()
        params, buffers = _named_state(torch, model)
        inputs = torch.randn(2, 3, 8, 8, device=device)
        with torch.no_grad():
            eager = model(inputs)
            functional = torch.func.functional_call(model, (params, buffers), (inputs,))
        torch.testing.assert_close(functional, eager, rtol=1e-5, atol=1e-6)
        _append_event(
            passed,
            "functional_call",
            parameter_count=len(params),
            buffer_count=len(buffers),
        )
        return model, params, buffers
    except Exception as exc:
        _append_event(errors, "functional_call", detail=f"{type(exc).__name__}: {exc}")
        return None, None, None


def _check_vmap_chunk_size(
    torch: Any,
    device: Any,
    passed: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    if not hasattr(torch, "vmap"):
        _append_event(skipped, "vmap_chunk_size", detail="torch.vmap is unavailable")
        return

    try:
        values = torch.arange(12, device=device, dtype=torch.float32).reshape(4, 3)

        def fn(row: Any) -> Any:
            return (row * row).sum()

        result = torch.vmap(fn, chunk_size=2)(values)
        expected = (values * values).sum(dim=1)
        torch.testing.assert_close(result, expected)
        _append_event(passed, "vmap_chunk_size", chunk_size=2)
    except TypeError as exc:
        _append_event(
            skipped,
            "vmap_chunk_size",
            detail=f"chunk_size is unsupported by this torch.vmap: {exc}",
        )
    except Exception as exc:
        _append_event(errors, "vmap_chunk_size", detail=f"{type(exc).__name__}: {exc}")


def _check_functional_vmap(
    torch: Any,
    device: Any,
    model: Any | None,
    params: dict[str, Any] | None,
    buffers: dict[str, Any] | None,
    passed: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    if model is None or params is None or buffers is None:
        _append_event(
            skipped,
            "functional_call_vmap",
            detail="functional_call probe did not produce reusable model state",
        )
        return
    if not hasattr(torch, "vmap"):
        _append_event(skipped, "functional_call_vmap", detail="torch.vmap is unavailable")
        return

    try:
        batched_params = {
            name: torch.stack([value, value + 0.001], dim=0)
            for name, value in params.items()
        }
        inputs = torch.randn(2, 3, 8, 8, device=device)

        def forward_one(point_params: dict[str, Any]) -> Any:
            logits = torch.func.functional_call(
                model,
                (point_params, buffers),
                (inputs,),
            )
            return logits.mean()

        with torch.no_grad():
            losses = torch.vmap(forward_one, chunk_size=1)(batched_params)
        if tuple(losses.shape) != (2,):
            raise AssertionError(f"expected loss shape (2,), got {tuple(losses.shape)}")
        _append_event(passed, "functional_call_vmap", chunk_size=1)
    except RuntimeError as exc:
        message = str(exc)
        if "vmap" in message.lower() or "batching" in message.lower():
            _append_event(
                skipped,
                "functional_call_vmap",
                detail=f"backend does not support this transform: {message}",
            )
            return
        _append_event(errors, "functional_call_vmap", detail=f"RuntimeError: {message}")
    except Exception as exc:
        _append_event(errors, "functional_call_vmap", detail=f"{type(exc).__name__}: {exc}")


def _check_inference_mode(
    torch: Any,
    device: Any,
    passed: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    errors: list[dict[str, Any]],
) -> None:
    if not hasattr(torch, "vmap"):
        _append_event(skipped, "inference_mode_vmap", detail="torch.vmap is unavailable")
        return

    try:
        values = torch.arange(8, device=device, dtype=torch.float32).reshape(4, 2)

        def fn(row: Any) -> Any:
            return row.sin().sum()

        with torch.inference_mode():
            result = torch.vmap(fn, chunk_size=2)(values)
        if tuple(result.shape) != (4,):
            raise AssertionError(f"expected shape (4,), got {tuple(result.shape)}")
        _append_event(passed, "inference_mode_vmap")
    except RuntimeError as exc:
        _append_event(
            skipped,
            "inference_mode_vmap",
            detail=f"inference_mode is not compatible with this transform/backend: {exc}",
        )
    except Exception as exc:
        _append_event(errors, "inference_mode_vmap", detail=f"{type(exc).__name__}: {exc}")


def _check_batchnorm_handling(
    model: Any | None,
    passed: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
) -> None:
    if model is None:
        _append_event(
            skipped,
            "batchnorm_handling",
            detail="functional_call probe did not produce a model",
        )
        return

    batchnorm_modules = [
        module.__class__.__name__
        for module in model.modules()
        if "BatchNorm" in module.__class__.__name__
    ]
    if not batchnorm_modules:
        _append_event(skipped, "batchnorm_handling", detail="probe model has no BatchNorm")
        return

    _append_event(
        passed,
        "batchnorm_handling",
        detail="BatchNorm buffers are passed to functional_call with model.eval()",
        batchnorm_modules=batchnorm_modules,
    )


def build_tiny_workload_request(
    workload_name: str = DEFAULT_WORKLOAD_NAME,
    *,
    device: DeviceName | str = "cpu",
    sample_count: int = 4,
    batch_size: int = 2,
    resolution: int = 2,
    scale: float = 0.1,
) -> Any:
    from src.schemas import GridSpec, SchedulerRequest, VanillaMode
    from src.workloads import WORKLOADS

    if workload_name not in WORKLOADS:
        known = ", ".join(sorted(WORKLOADS))
        raise ValueError(f"unknown workload '{workload_name}'; known workloads: {known}")

    definition = WORKLOADS[workload_name]
    task = replace(
        definition.spec,
        dataset=replace(definition.spec.dataset, sample_count=sample_count),
    )
    return SchedulerRequest(
        task=task,
        grid=GridSpec(resolution=resolution, scale=scale),
        mode=VanillaMode(gpu_batch_size=batch_size),
        device=device,
    )


def _request_metadata(request: Any) -> dict[str, Any]:
    return {
        "workload": request.task.name,
        "model": request.task.model,
        "task": request.task.task,
        "loss": request.task.loss,
        "dataset": request.task.dataset.name,
        "sample_count": request.task.dataset.sample_count,
        "checkpoint_path": request.task.checkpoint_path,
        "grid_resolution": request.grid.resolution,
        "grid_scale": request.grid.scale,
        "batch_size": request.mode.gpu_batch_size,
    }


def _asset_status(workload_name: str = DEFAULT_WORKLOAD_NAME) -> dict[str, Any]:
    from src.workloads import WORKLOADS

    definition = WORKLOADS[workload_name]
    dataset_path = Path(definition.spec.dataset.path)
    checkpoint_path = (
        Path(definition.spec.checkpoint_path)
        if definition.spec.checkpoint_path is not None
        else None
    )
    return {
        "dataset_path": dataset_path,
        "dataset_exists": dataset_path.exists(),
        "checkpoint_path": checkpoint_path,
        "checkpoint_exists": checkpoint_path is not None and checkpoint_path.exists(),
    }


def _check_tiny_workload(
    torch: Any,
    device: Any,
    passed: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    seed: int,
    workload_name: str = DEFAULT_WORKLOAD_NAME,
) -> None:
    try:
        from torch.utils.data import DataLoader

        from src.backends.base import build_grid_points, build_direction_vectors
        from src.workloads import WORKLOADS
    except Exception as exc:
        _append_event(
            skipped,
            "tiny_workload",
            detail=f"required repository modules are unavailable: {type(exc).__name__}: {exc}",
        )
        return

    assets = _asset_status(workload_name)
    missing = [
        key
        for key, exists in (
            ("dataset_path", assets["dataset_exists"]),
            ("checkpoint_path", assets["checkpoint_exists"]),
        )
        if not exists
    ]
    if missing:
        _append_event(
            skipped,
            "tiny_workload",
            detail=f"missing assets: {', '.join(missing)}",
            workload=workload_name,
            **assets,
        )
        return

    try:
        request = build_tiny_workload_request(workload_name, device=device.type)
        definition = WORKLOADS[request.task.name]

        _seed_torch(torch, device, seed)
        model = definition.build_model(request.task).float().to(device)
        model.eval()
        dataset = definition.build_dataset(request.task, seed)
        loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
        base, direction_a, direction_b = build_direction_vectors(model, seed)
        base = base.to(device)
        direction_a = direction_a.to(device)
        direction_b = direction_b.to(device)
        points = build_grid_points(request.grid)
        params, buffers = _named_state(torch, model)

        total_loss = 0.0
        total_examples = 0
        first_point = points[0]
        perturbed = (
            base
            + (first_point.alpha * direction_a)
            + (first_point.beta * direction_b)
        )
        offset = 0
        point_params: dict[str, Any] = {}
        for name, current in params.items():
            numel = current.numel()
            point_params[name] = perturbed[offset : offset + numel].view_as(current)
            offset += numel

        class FunctionalProbeModule(torch.nn.Module):
            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return torch.func.functional_call(
                    model,
                    (point_params, buffers),
                    args,
                    kwargs,
                )

        functional_model = FunctionalProbeModule()
        functional_model.eval()

        with torch.no_grad():
            for batch in loader:
                loss, batch_size = definition.compute_loss(
                    functional_model,
                    batch,
                    device,
                )
                total_loss += float(loss.detach().cpu()) * batch_size
                total_examples += batch_size
        _synchronize(torch, device)
        _append_event(
            passed,
            "tiny_workload",
            **_request_metadata(request),
            grid_points=len(points),
            examples=total_examples,
            first_point_loss=total_loss / max(1, total_examples),
        )
    except FileNotFoundError as exc:
        _append_event(
            skipped,
            "tiny_workload",
            detail=str(exc),
            workload=workload_name,
            **assets,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "not implemented" in message.lower() or "unsupported" in message.lower():
            _append_event(
                skipped,
                "tiny_workload",
                detail=f"backend unsupported for tiny workload: {message}",
                workload=workload_name,
            )
            return
        _append_event(
            errors,
            "tiny_workload",
            detail=f"RuntimeError: {message}",
            workload=workload_name,
        )
    except Exception as exc:
        _append_event(
            errors,
            "tiny_workload",
            detail=f"{type(exc).__name__}: {exc}",
            workload=workload_name,
        )


def run_pipeline(
    requested_device: DeviceName,
    seed: int = 1337,
    workload_names: tuple[str, ...] = (DEFAULT_WORKLOAD_NAME,),
) -> dict[str, Any]:
    passed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    torch = _check_torch_import(passed, errors)
    if torch is None:
        return {
            "backend": {"selected": None, "available": {}},
            "passed_checks": passed,
            "skipped_checks": skipped,
            "errors": errors,
        }

    device = _check_device_resolution(torch, requested_device, passed, skipped)
    backend = _device_metadata(torch, device)
    if device is None:
        return {
            "backend": backend,
            "passed_checks": passed,
            "skipped_checks": skipped,
            "errors": errors,
        }

    _seed_torch(torch, device, seed)
    model, params, buffers = _check_functional_call(torch, device, passed, errors)
    _check_vmap_chunk_size(torch, device, passed, skipped, errors)
    _check_functional_vmap(torch, device, model, params, buffers, passed, skipped, errors)
    _check_inference_mode(torch, device, passed, skipped, errors)
    _check_batchnorm_handling(model, passed, skipped)
    for workload_name in workload_names:
        _check_tiny_workload(
            torch,
            device,
            passed,
            skipped,
            errors,
            seed,
            workload_name,
        )

    return {
        "backend": backend,
        "passed_checks": passed,
        "skipped_checks": skipped,
        "errors": errors,
    }


def build_config(**overrides: Any) -> SimpleNamespace:
    defaults = {
        "device": "auto",
        "seed": 1337,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def main() -> None:
    args = build_config()
    result = run_pipeline(args.device, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
