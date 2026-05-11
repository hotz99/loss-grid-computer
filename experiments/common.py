from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from experiments.functional_eval_experiments import DEFAULT_FUNCTIONAL_EVAL_WORKLOADS
from src.workloads import WORKLOADS


def configured_workloads(config: SimpleNamespace) -> list[str]:
    names = list(getattr(config, "mltask_workloads", None) or [])
    if not names:
        names = list(DEFAULT_FUNCTIONAL_EVAL_WORKLOADS)
    return names


def workload_unavailable_reason(name: str) -> str | None:
    definition = WORKLOADS.get(name)
    if definition is None:
        return f"workload is not registered: {name}"

    dataset_path = Path(definition.spec.dataset.path)
    if not dataset_path.exists():
        return f"dataset asset is missing: {dataset_path}"

    checkpoint_path = definition.spec.checkpoint_path
    if checkpoint_path is not None and not Path(checkpoint_path).exists():
        return f"checkpoint asset is missing: {checkpoint_path}"

    return None


def workload_metadata(name: str, sample_count: int | None = None) -> dict[str, Any]:
    definition = WORKLOADS.get(name)
    if definition is None:
        return {"workload_name": name, "registered": False}

    spec = definition.spec
    dataset = spec.dataset
    if sample_count is not None:
        dataset = replace(dataset, sample_count=sample_count)
    return {
        "workload_name": spec.name,
        "registered": True,
        "model": spec.model,
        "task": spec.task,
        "loss": spec.loss,
        "dataset": asdict(dataset),
        "checkpoint_path": spec.checkpoint_path,
    }


def unavailable_payload(
    name: str,
    reason: str,
    sample_count: int | None = None,
) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
        "workload": workload_metadata(name, sample_count),
    }


def shared_artifact_key(
    *,
    node_type: str,
    workload_name: str,
    checkpoint_path: str | None,
    sample_count: int | None,
    grid_resolution: int,
    grid_scale: float,
    device: str,
    seed: int,
    gpu_batch_size: int,
    slowdown_factor: float,
) -> str:
    payload = {
        "node_type": node_type,
        "workload_name": workload_name,
        "checkpoint_path": checkpoint_path,
        "sample_count": sample_count,
        "grid_resolution": grid_resolution,
        "grid_scale": grid_scale,
        "device": device,
        "seed": seed,
        "gpu_batch_size": gpu_batch_size,
        "slowdown_factor": slowdown_factor,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def get_shared_artifact(shared_state: dict[str, Any], key: str) -> Any | None:
    return shared_state.get("_artifact_objects", {}).get(key)


def put_shared_artifact(
    shared_state: dict[str, Any],
    key: str,
    value: Any,
    summary: dict[str, Any],
) -> None:
    shared_state.setdefault("_artifact_objects", {})[key] = value
    shared_state.setdefault("artifacts", {})[key] = summary
