"""Shared fixtures: the tiny smoke configuration."""

from pathlib import Path

import pytest

from ricor.config import load_config

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / "configs" / "experiments" / "smoke.yaml"
METHODS = ROOT / "configs" / "methods"


@pytest.fixture
def smoke_cfg():
    return load_config(SMOKE)


def method_cfg(name: str, **extra):
    import yaml

    path = METHODS / f"{name}.yaml" if (METHODS / f"{name}.yaml").is_file() else METHODS / "ablations" / f"{name}.yaml"
    with open(path, encoding="utf-8") as fh:
        method = yaml.safe_load(fh)
    return load_config(SMOKE, {"method": method, **extra})
