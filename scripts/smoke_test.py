"""End-to-end smoke test: scenarios -> train -> evaluate -> infer -> aggregate on a tiny CPU config.

Validates the software pipeline only. The numbers it prints are NOT research results.

    python scripts/smoke_test.py [--out runs/smoke_check] [--methods ricor hpsf]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ricor.cli import aggregate_main, evaluate_main, infer_main, scenarios_main, train_main
from ricor.utils import read_json

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "smoke.yaml"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=str(ROOT / "runs" / "smoke_check"))
    p.add_argument("--methods", nargs="+", default=["ricor"])
    args = p.parse_args()
    out = Path(args.out)
    t0 = time.perf_counter()

    assert scenarios_main(["--config", str(CONFIG), "--out", str(out / "scenarios")]) == 0
    for method in args.methods:
        method_cfg = ROOT / "configs" / "methods" / f"{method}.yaml"
        if not method_cfg.is_file():
            print(f"unknown method {method}", file=sys.stderr)
            return 2
        assert (
            train_main(["--config", str(CONFIG), "--method", str(method_cfg), "--set", f"output_dir={out.as_posix()}"])
            == 0
        )
        ckpt = out / "smoke" / method / "seed0" / "checkpoint.pt"
        assert ckpt.is_file(), f"missing checkpoint {ckpt}"
        assert evaluate_main(["--checkpoint", str(ckpt)]) == 0
        assert infer_main(["--checkpoint", str(ckpt), "--condition", "C2"]) == 0
        ev = read_json(ckpt.with_name("eval.json"))
        missing = [c for c in ev["conditions"] if "apr" not in ev["conditions"][c]["summary"]]
        assert not missing, f"conditions without APR: {missing}"
        print(
            f"[smoke] {method}: clean APR={ev['conditions']['clean']['summary']['apr']:.1f} "
            f"(pipeline check only, not a result)"
        )
    assert aggregate_main(["--root", str(out / "smoke")]) == 0
    print(f"[smoke] OK in {time.perf_counter() - t0:.1f}s; outputs under {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
