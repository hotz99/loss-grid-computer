from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.backends.base import resolve_device
from src.workloads import WORKLOADS


def asset_summary() -> dict[str, Any]:
    assets: dict[str, Any] = {}
    for name, definition in WORKLOADS.items():
        dataset_path = Path(definition.spec.dataset.path)
        checkpoint_path = (
            None
            if definition.spec.checkpoint_path is None
            else Path(definition.spec.checkpoint_path)
        )
        assets[name] = {
            "dataset_path": str(dataset_path),
            "dataset_exists": dataset_path.exists(),
            "checkpoint_path": None
            if checkpoint_path is None
            else str(checkpoint_path),
            "checkpoint_exists": (
                None if checkpoint_path is None else checkpoint_path.exists()
            ),
        }
    return assets


def run(
    config: SimpleNamespace,
    output_dir: Path,
    shared_state: dict[str, Any],
    *,
    platform_summary,
) -> dict[str, Any]:
    del output_dir
    del shared_state
    resolved_device = resolve_device(config.device)
    platform = platform_summary(config.device, resolved_device)
    gpu = platform.get("gpu") or {}
    cpu = platform.get("cpu") or {}
    memory = platform.get("memory") or {}
    assets = asset_summary()
    return {
        "status": "completed",
        "platform": platform,
        "assets": assets,
        "record": {
            "status": "completed",
            "resolved_device": platform.get("resolved_device"),
            "gpu_model": gpu.get("name"),
            "vram_bytes": gpu.get("total_memory_bytes"),
            "ram_bytes": memory.get("total_system_memory_bytes"),
            "cpu_logical_cores": cpu.get("logical_cores"),
            "asset_count": len(assets),
            "missing_asset_count": sum(
                1
                for item in assets.values()
                if not item["dataset_exists"]
                or item["checkpoint_exists"] is False
            ),
        },
    }
