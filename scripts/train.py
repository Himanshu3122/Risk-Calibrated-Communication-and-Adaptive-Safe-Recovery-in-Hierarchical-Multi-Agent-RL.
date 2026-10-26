"""Train RiCoR or a baseline. See `python scripts/train.py --help`."""

import sys

from ricor.cli import train_main

if __name__ == "__main__":
    sys.exit(train_main())
