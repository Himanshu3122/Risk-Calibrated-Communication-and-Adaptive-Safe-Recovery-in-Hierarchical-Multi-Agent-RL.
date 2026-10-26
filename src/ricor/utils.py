"""Seeding, logging, and run-directory helpers."""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

LOGGER_NAME = "ricor"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a module logger with a single structured stream handler."""
    logger = logging.getLogger(name)
    if not logging.getLogger(LOGGER_NAME).handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s | %(message)s"))
        root = logging.getLogger(LOGGER_NAME)
        root.addHandler(handler)
        root.setLevel(os.environ.get("RICOR_LOG_LEVEL", "INFO").upper())
    return logger


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch; optionally request deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def safe_child(root: str | Path, *parts: str) -> Path:
    """Join ``parts`` under ``root`` and refuse paths that escape it."""
    base = Path(root).resolve()
    target = base.joinpath(*parts).resolve()
    if base != target and base not in target.parents:
        raise ValueError(f"path {target} escapes {base}")
    return target


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=5)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def environment_metadata() -> dict[str, Any]:
    """Versions and platform information recorded with every run."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "git_sha": _git_sha(),
    }


def make_run_dir(output_dir: str | Path, name: str, method: str, seed: int) -> Path:
    """Create ``<output_dir>/<name>/<method>/seed<seed>`` and return it."""
    run_dir = safe_child(output_dir, name, method, f"seed{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, payload: Any) -> None:
    """Write JSON with stable formatting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=_json_default)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"not JSON serializable: {type(obj)}")
