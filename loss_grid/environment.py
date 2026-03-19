from __future__ import annotations

import os
import platform
import socket
import subprocess
from typing import Any, Dict

import torch


def _safe_command(command: str) -> str:
    try:
        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        output = completed.stdout.strip() or completed.stderr.strip()
        return output
    except Exception:
        return ""


def capture_environment() -> Dict[str, Any]:
    devices = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "multi_processor_count": props.multi_processor_count,
                }
            )

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_devices": devices,
        "mpi_version": _safe_command("mpirun --version"),
        "nvidia_smi": _safe_command("nvidia-smi --query-gpu=name,driver_version --format=csv,noheader"),
    }
