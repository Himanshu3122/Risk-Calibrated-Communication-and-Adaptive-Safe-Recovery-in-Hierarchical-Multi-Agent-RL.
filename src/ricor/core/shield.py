"""One-step safety shield (paper Eq. 11, Proposition 1).

The predicate g_hat(o, a) <= 0 is evaluated from the crew's local patch only.
An action is unsafe if its destination cell is burning, or the destination or any of its
four neighbors has spread probability above the entrapment threshold, or the destination
lies within ``drop_radius`` of a drop target the crew knows about. Proposition 1's
guarantee holds only if this local view over-approximates the true unsafe set and the
fallback action is safe; neither is guaranteed in HierFire, so violations remain possible.
"""

from __future__ import annotations

import numpy as np

from ricor.envs.hierfire import MOVES, N_CREW_ACTIONS

INTENSITY, SPREAD = 0, 1


def _destination(action: int) -> tuple[int, int]:
    return MOVES.get(action, (0, 0))


def unsafe_actions(patch: np.ndarray, drop_mask: np.ndarray, threshold: float, drop_radius: int) -> np.ndarray:
    """g_hat > 0 for each crew action. ``patch``: [3, P, P]; ``drop_mask``: [P, P] bool."""
    P = patch.shape[-1]
    c = P // 2
    hot = patch[SPREAD] > threshold
    burning = patch[INTENSITY] > 0
    unsafe = np.zeros(N_CREW_ACTIONS, dtype=bool)
    drops = np.argwhere(drop_mask)
    for a in range(N_CREW_ACTIONS):
        dy, dx = _destination(a)
        y, x = c + dy, c + dx
        cells = [(y, x), (y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        bad = any(0 <= yy < P and 0 <= xx < P and hot[yy, xx] for yy, xx in cells)
        bad |= 0 <= y < P and 0 <= x < P and bool(burning[y, x])
        if len(drops):
            bad |= bool((np.abs(drops - np.array([y, x])).max(axis=1) <= drop_radius).any())
        unsafe[a] = bad
    return unsafe


def fallback_action(patch: np.ndarray) -> int:
    """a_fb: move one step away from the hottest visible cell (stay if nothing burns)."""
    P = patch.shape[-1]
    c = P // 2
    heat = patch[INTENSITY] + patch[SPREAD]
    if heat.max() <= 0:
        return 0
    hy, hx = np.unravel_index(int(np.argmax(heat)), heat.shape)
    best, best_d = 0, -1.0
    for a in (0, 1, 2, 3, 4):
        dy, dx = _destination(a)
        d = float(np.hypot(c + dy - hy, c + dx - hx))
        if d > best_d:
            best, best_d = a, d
    return best


def shield(
    proposed: int, probs: np.ndarray, patch: np.ndarray, drop_mask: np.ndarray, threshold: float, drop_radius: int
) -> tuple[int, bool]:
    """Eq. (11): keep a safe proposal, else the most probable safe action, else a_fb.

    Returns the executed action and whether the shield intervened.
    """
    unsafe = unsafe_actions(patch, drop_mask, threshold, drop_radius)
    if not unsafe[proposed]:
        return proposed, False
    safe = np.flatnonzero(~unsafe)
    if len(safe):
        return int(safe[np.argmax(probs[safe])]), True
    return fallback_action(patch), True
