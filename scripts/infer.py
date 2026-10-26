"""Run one episode with a checkpoint and write a step trace. See `python scripts/infer.py --help`."""

import sys

from ricor.cli import infer_main

if __name__ == "__main__":
    sys.exit(infer_main())
