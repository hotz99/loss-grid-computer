from __future__ import annotations

import os
import platform
from typing import Any

from experiments.workloads import WORKLOADS


def run(device: str) -> dict[str, Any]:
    platform_summary = {
        "requested_device": device,
        "host_os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
        "cpu": {
            "logical_cores": os.cpu_count(),
            "model": platform.processor() or None,
        },
        "env_threads": {
            key: os.environ.get(key)
            for key in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
    }
    assets: dict[str, Any] = {
        name: {
            "dataset": workload.dataset,
            "model": workload.model,
            "task": workload.task,
            "loss": workload.loss,
        }
        for name, workload in WORKLOADS.items()
    }
    record = {
        "status": "completed",
        "requested_device": device,
        "cpu_logical_cores": platform_summary["cpu"]["logical_cores"],
        "workload_count": len(assets),
    }
    return {
        "status": "completed",
        "platform": platform_summary,
        "assets": assets,
        "record": record,
    }
