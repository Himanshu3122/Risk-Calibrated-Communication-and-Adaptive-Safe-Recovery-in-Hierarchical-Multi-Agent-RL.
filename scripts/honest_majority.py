"""Exact honest-majority analysis of the verification vote on the HierFire region graph.

For an honest sender j whose n_j neighbors are queried, the vote corroborates j only if a strict
majority of the received replies is consistent. Honest neighbors never contradict an honest, fresh,
unaltered report (their border view is a subset of j's region), so without reply loss the vote can
fail only if compromised neighbors make up at least half of the replies. With k compromised agents
drawn uniformly from the other N - 1 regions, this script computes, averaged over senders,
P(#compromised neighbors >= n_j / 2), i.e., the probability that the adversary can block
corroboration of an honest sender when every reply arrives.

Usage: python scripts/honest_majority.py [--rows 4 --cols 2]
"""

from __future__ import annotations

import argparse
from math import ceil, comb


def neighbors(r: int, rows: int, cols: int) -> list[int]:
    ri, ci = divmod(r, cols)
    out = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        a, b = ri + dr, ci + dc
        if 0 <= a < rows and 0 <= b < cols:
            out.append(a * cols + b)
    return out


def block_probability(n_nb: int, others: int, k: int) -> float:
    """P(at least ceil(n_nb / 2) of the n_nb neighbors are among k compromised of ``others``)."""
    need = ceil(n_nb / 2)
    total = comb(others, k)
    return sum(comb(n_nb, x) * comb(others - n_nb, k - x) for x in range(need, min(n_nb, k) + 1)) / total


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, default=4)
    p.add_argument("--cols", type=int, default=2)
    a = p.parse_args()
    n = a.rows * a.cols
    degs = [len(neighbors(r, a.rows, a.cols)) for r in range(n)]
    print(f"regions={n}, neighbor counts={degs}")
    for k in range(1, 5):
        probs = [block_probability(d, n - 1, k) for d in degs]
        print(f"k={k}: mean P(block honest sender)={sum(probs) / n:.3f}  "
              f"(2-neighbor senders {probs[degs.index(2)]:.3f}, 3-neighbor senders {probs[degs.index(3)]:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
