"""Aggregate eval.json files across methods and seeds. See `python scripts/aggregate.py --help`."""

import sys

from ricor.cli import aggregate_main

if __name__ == "__main__":
    sys.exit(aggregate_main())
