"""Evaluate a checkpoint under the paper fault conditions. See `python scripts/evaluate.py --help`."""

import sys

from ricor.cli import evaluate_main

if __name__ == "__main__":
    sys.exit(evaluate_main())
