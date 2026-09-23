"""Experiment metadata and structured logging helpers."""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def get_git_commit(project_root: str | Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=project_root,
            capture_output=True, check=False, text=True,
        )
    except OSError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None



def collect_runtime_metadata(project_root: str | Path | None = None) -> dict[str, Any]:
    cuda_devices = []
    if torch.cuda.is_available():
        for device_index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(device_index)
            cuda_devices.append(
                {
                    "index": device_index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                }
            )

    slurm_keys = (
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "SLURM_NODELIST",
        "SLURM_NTASKS",
        "SLURM_PROCID",
        "SLURM_LOCALID",
    )
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit(project_root),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_devices": cuda_devices,
        "slurm": {key: os.environ[key] for key in slurm_keys if key in os.environ},
    }


def configure_logging(log_path: str | Path, enabled: bool = True) -> None:
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    if not enabled:
        root_logger.addHandler(logging.NullHandler())
        root_logger.setLevel(logging.CRITICAL)
        return

    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(logging.INFO)


def write_json(path: str | Path, value: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)


def update_json(path: str | Path, updates: dict[str, Any]) -> None:
    output_path = Path(path)
    current = json.loads(output_path.read_text(encoding="utf-8"))
    current.update(updates)
    write_json(output_path, current)
