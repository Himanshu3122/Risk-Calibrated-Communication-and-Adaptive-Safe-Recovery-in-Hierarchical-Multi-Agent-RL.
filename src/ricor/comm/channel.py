"""Fault-prone channel with observable metadata and hidden fault labels (paper Eq. 2).

Node 0 is the coordinator; regional agent ``r`` is node ``r + 1``. Every directed link
keeps a Gilbert-Elliott state, an in-flight queue, and the newest delivered message.
The receiver observes only the metadata ``eta = (arrived, age, gap)`` and the content.
The labels ``y_ch`` (fresh delivery) and ``y_cr`` (unaltered content) are returned for
supervision and evaluation and must never be fed to a policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ricor.comm.faults import FaultSpec

UPLINK, PEER, DOWNLINK = 0, 1, 2
FAULT_NONE, FAULT_CORRUPT, FAULT_STALE, FAULT_MISLEAD = 0, 1, 2, 3
HEADER_BYTES = 16
THREAT_FIELDS = (0, 1, 2, 5)  # burning share, intensity, max spread, threatened assets
AGE_CAP = 50


@dataclass
class Delivery:
    """State of every link after one transmission round."""

    arrived: np.ndarray  # [L] bool, a message arrived this step
    age: np.ndarray  # [L] int, age of the newest held message (AGE_CAP if none)
    gap: np.ndarray  # [L] bool, sequence numbers were skipped
    has_content: np.ndarray  # [L] bool
    content: np.ndarray  # [L, D] newest held content
    seq: np.ndarray  # [L] int, send time of held content (-1 if none)
    y_ch: np.ndarray  # [L] float, held message delivered fresh this step
    y_cr: np.ndarray  # [L] float, held content unaltered
    fault_type: np.ndarray  # [L] int, FAULT_* code of held content
    bytes_sent: int


class Channel:
    """Directed links among coordinator and regional agents."""

    def __init__(self, n_regions: int, peers: list[list[int]], payload_dim: int, goal_dim: int) -> None:
        self.n_regions = n_regions
        self.payload_dim = payload_dim
        self.goal_dim = goal_dim
        links: list[tuple[int, int, int]] = []
        for r in range(n_regions):
            links.append((r + 1, 0, UPLINK))
        for r in range(n_regions):
            for q in peers[r]:
                links.append((r + 1, q + 1, PEER))
        for r in range(n_regions):
            links.append((0, r + 1, DOWNLINK))
        self.links = links
        self.n_links = len(links)
        self.uplink_idx = np.array([i for i, link in enumerate(links) if link[2] == UPLINK])
        self.downlink_idx = np.array([i for i, link in enumerate(links) if link[2] == DOWNLINK])
        self.peer_idx = {(s - 1, d - 1): i for i, (s, d, k) in enumerate(links) if k == PEER}
        self.dim = max(payload_dim, goal_dim)
        self.spec = FaultSpec()
        self.window: tuple[int, int] | None = None
        self.rng = np.random.default_rng(0)
        self.reset(FaultSpec(), seed=0)

    def reset(self, spec: FaultSpec, seed: int, window: tuple[int, int] | None = None) -> None:
        """Start an episode with fault ``spec``, active only inside ``window`` if given."""
        self.spec = spec
        self.window = window
        self.rng = np.random.default_rng(seed)
        L, D = self.n_links, self.dim
        self.ge_bad = np.zeros(L, dtype=bool)
        self.queues: list[list[tuple[int, int, np.ndarray, int]]] = [[] for _ in range(L)]
        self.held = np.zeros((L, D), dtype=np.float32)
        self.held_seq = np.full(L, -1, dtype=np.int64)
        self.held_type = np.zeros(L, dtype=np.int64)
        self.last_seq = np.full(L, -1, dtype=np.int64)
        k = min(spec.compromised, self.n_regions)
        self.compromised = set(self.rng.choice(self.n_regions, size=k, replace=False).tolist()) if k else set()
        n_mh = int(round(spec.multihop_frac * self.n_regions))
        self.multihop = set(self.rng.choice(self.uplink_idx, size=n_mh, replace=False).tolist()) if n_mh else set()
        self.history: list[np.ndarray] = []
        # Verification replies draw from their own stream so that queries never perturb the
        # loss process (or the Gilbert-Elliott state) of ordinary traffic.
        self.reply_rng = np.random.default_rng(seed + 7_919)
        self.attack_start: int | None = None

    def active(self, t: int) -> bool:
        return self.window is None or self.window[0] <= t < self.window[1]

    # ------------------------------------------------------------------ content faults
    def mislead(self, report: np.ndarray, eps: float) -> np.ndarray:
        """Misleading report: push threat fields toward the opposite extreme by <= eps (assumption A9)."""
        out = report.copy()
        idx = list(THREAT_FIELDS)
        out[idx] = np.clip(out[idx] + eps * np.sign(0.5 - out[idx]), 0.0, 1.0)
        return out

    def mislead_eps(self, t: int) -> float:
        """Perturbation budget at step ``t``; a slow-drift adversary ramps it up over ``mislead_ramp`` steps."""
        s = self.spec
        if s.mislead_ramp <= 0:
            return s.mislead_eps
        if self.attack_start is None:
            self.attack_start = t
        return s.mislead_eps * min(1.0, (t - self.attack_start + 1) / s.mislead_ramp)

    def _transform(self, sender_region: int, payload: np.ndarray, t: int) -> tuple[np.ndarray, int]:
        s, rng = self.spec, self.rng
        if s.compromised and sender_region in self.compromised:
            return self.mislead(payload, self.mislead_eps(t)), FAULT_MISLEAD
        if s.stale_rate and rng.random() < s.stale_rate and len(self.history) > s.stale_min:
            age = int(rng.integers(s.stale_min, min(s.stale_max, len(self.history) - 1) + 1))
            return self.history[-1 - age][sender_region].copy(), FAULT_STALE
        if s.corrupt_rate and rng.random() < s.corrupt_rate:
            out = payload.copy()
            mask = rng.random(out.shape) < s.corrupt_fields
            out[mask] += rng.normal(0.0, s.corrupt_sigma, size=int(mask.sum()))
            return out, FAULT_CORRUPT
        return payload, FAULT_NONE

    # ------------------------------------------------------------------ transmission
    def link_survives(self, link: int, faulty: bool) -> bool:
        """Sample loss for one transmission on ``link`` and advance its Gilbert-Elliott state."""
        s, rng = self.spec, self.rng
        if not faulty:
            return True
        if s.ge_p_gb > 0:
            if self.ge_bad[link]:
                self.ge_bad[link] = rng.random() >= s.ge_p_bg
            else:
                self.ge_bad[link] = rng.random() < s.ge_p_gb
            if self.ge_bad[link] and rng.random() < s.ge_loss_bad:
                return False
        if s.loss_p and rng.random() < s.loss_p:
            return False
        if link in self.multihop and rng.random() < s.multihop_loss:
            return False
        return True

    def transmit(self, t: int, reports: np.ndarray, goals: np.ndarray) -> Delivery:
        """Send this step's reports (uplink and peer) and goals (downlink), then deliver."""
        faulty = self.active(t) and not self.spec.is_clean
        self.history.append(reports.copy())
        if len(self.history) > AGE_CAP:
            self.history.pop(0)
        bytes_sent = 0
        for li, (src, dst, kind) in enumerate(self.links):
            if kind == DOWNLINK:
                payload, ftype = goals[dst - 1], FAULT_NONE
            else:
                payload, ftype = reports[src - 1], FAULT_NONE
                if faulty:
                    payload, ftype = self._transform(src - 1, payload, t)
            bytes_sent += HEADER_BYTES + 4 * len(payload)
            if not self.link_survives(li, faulty):
                continue
            delay = 0
            if faulty and self.spec.delay_max > 0:
                delay = int(self.rng.integers(self.spec.delay_min, self.spec.delay_max + 1))
            if faulty and li in self.multihop:
                delay += int(self.rng.integers(1, self.spec.multihop_delay + 2))
            buf = np.zeros(self.dim, dtype=np.float32)
            buf[: len(payload)] = payload
            self.queues[li].append((t + delay, t, buf, ftype))

        L = self.n_links
        arrived = np.zeros(L, dtype=bool)
        gap = np.zeros(L, dtype=bool)
        for li in range(L):
            ready = [m for m in self.queues[li] if m[0] <= t]
            if not ready:
                continue
            self.queues[li] = [m for m in self.queues[li] if m[0] > t]
            newest = max(ready, key=lambda m: m[1])
            if newest[1] > self.held_seq[li]:
                arrived[li] = True
                gap[li] = self.last_seq[li] >= 0 and newest[1] - self.last_seq[li] > 1
                self.last_seq[li] = newest[1]
                self.held[li] = newest[2]
                self.held_seq[li] = newest[1]
                self.held_type[li] = newest[3]
        has = self.held_seq >= 0
        age = np.where(has, t - self.held_seq, AGE_CAP).clip(0, AGE_CAP)
        y_ch = (arrived & (age <= 1)).astype(np.float32)
        y_cr = (self.held_type == FAULT_NONE).astype(np.float32)
        return Delivery(
            arrived=arrived,
            age=age,
            gap=gap,
            has_content=has,
            content=self.held.copy(),
            seq=self.held_seq.copy(),
            y_ch=y_ch,
            y_cr=y_cr,
            fault_type=self.held_type.copy(),
            bytes_sent=bytes_sent,
        )

    def verification_reply(self, t: int, neighbor_region: int, border_view: np.ndarray) -> np.ndarray | None:
        """A neighbor's reply to a verification query (assumption A12).

        ``border_view`` is what the neighbor itself observes of the suspicious sender's region
        (``HierFireEnv.border_view``): ``[burning cells, max spread]`` in the shared border band.
        The reply travels over the neighbor's uplink and is lost with the probability that link
        currently has, sampled from a separate stream without advancing the link's
        Gilbert-Elliott state. A compromised neighbor inflates its reported spread.
        """
        s = self.spec
        faulty = self.active(t) and not s.is_clean
        link = int(self.uplink_idx[neighbor_region])
        if faulty:
            rng = self.reply_rng
            if self.ge_bad[link] and rng.random() < s.ge_loss_bad:
                return None
            if s.loss_p and rng.random() < s.loss_p:
                return None
            if link in self.multihop and rng.random() < s.multihop_loss:
                return None
        reply = border_view.astype(np.float32).copy()
        if faulty and neighbor_region in self.compromised:
            eps = self.mislead_eps(t)
            reply[1] = float(np.clip(reply[1] + eps * np.sign(0.5 - reply[1]), 0.0, 1.0))
        return reply
