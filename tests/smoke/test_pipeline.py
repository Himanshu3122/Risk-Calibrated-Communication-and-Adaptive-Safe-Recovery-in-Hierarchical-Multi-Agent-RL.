"""End-to-end CLI pipeline on the smoke configuration (software check, not research results)."""

import pytest

from ricor.cli import aggregate_main, evaluate_main, infer_main, scenarios_main, train_main
from ricor.utils import read_json

from ..conftest import METHODS, SMOKE


@pytest.mark.smoke
def test_cli_pipeline(tmp_path):
    out = tmp_path.as_posix()
    assert scenarios_main(["--config", str(SMOKE), "--out", f"{out}/scenarios"]) == 0
    assert (
        train_main(["--config", str(SMOKE), "--method", str(METHODS / "ricor.yaml"), "--set", f"output_dir={out}"]) == 0
    )
    ckpt = tmp_path / "smoke" / "ricor" / "seed0" / "checkpoint.pt"
    assert ckpt.is_file()
    assert evaluate_main(["--checkpoint", str(ckpt)]) == 0
    ev = read_json(ckpt.with_name("eval.json"))
    assert set(ev["conditions"]) == {"clean", "F1", "F2", "F3", "F4", "F5", "F6", "C1", "C2", "C3", "U1", "U2", "U3"}
    assert set(ev["loss_sweep"]) == {"0.0", "0.3"}
    assert "recovery_steps" in ev["recovery"] and "pooled" in ev["calibration"]
    assert infer_main(["--checkpoint", str(ckpt), "--condition", "U3"]) == 0
    assert (ckpt.parent / "infer_U3.json").is_file()
    assert aggregate_main(["--root", str(tmp_path / "smoke")]) == 0
    assert (tmp_path / "smoke" / "summary.md").is_file()


def test_cli_rejects_bad_input(tmp_path):
    assert train_main(["--config", str(SMOKE), "--set", "method.modes=sometimes"]) == 2
