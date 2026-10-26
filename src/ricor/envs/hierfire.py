"""HierFire: hierarchical wildfire-suppression environment (paper Sec. VI-A).

The paper specifies the grid, region layout, crews, drops, coordinator period, episode
length, asset counts, and the two safety predicates. The fire model follows the
cellular-automaton structure of Alexandridis et al. (2008): spread probability depends
on fuel, wind alignment, slope, and source intensity. Its constants are not given in the
paper and are documented as assumption A1 in ``docs/assumptions.md``.

Crew action set (assumption A2): 0 stay, 1 north, 2 south, 3 east, 4 west,
5 suppress, 6 firebreak, 7 request resources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ricor.config import EnvConfig

N_CREW_ACTIONS = 8
REPORT_DIM = 16
PATCH_CHANNELS = 3  # fire intensity, spread probability, fuel
MOVES = {1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}
_NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


@dataclass
class StepInfo:
    """Per-step quantities returned alongside the reward."""

    crew_costs: np.ndarray  # [N, C] safety cost events per crew this step
    entrapment_events: int
    drop_events: int
    newly_ignited: int
    asset_value_lost: float


def _smooth(field: np.ndarray, passes: int = 3) -> np.ndarray:
    out = field.copy()
    for _ in range(passes):
        padded = np.pad(out, 1, mode="edge")
        out = (
            sum(
                padded[1 + dy : 1 + dy + out.shape[0], 1 + dx : 1 + dx + out.shape[1]]
                for dy in (-1, 0, 1)
                for dx in (-1, 0, 1)
            )
            / 9.0
        )
    return out


class HierFireEnv:
    """Single HierFire episode simulator with numpy state."""

    def __init__(self, cfg: EnvConfig) -> None:
        cfg.validate()
        self.cfg = cfg
        self.n_regions = cfg.n_regions
        self.n_crews = cfg.crews_per_region
        self.region_h = cfg.height // cfg.region_rows
        self.region_w = cfg.width // cfg.region_cols
        self.patch_size = 2 * cfg.patch_radius + 1
        self.patch_dim = PATCH_CHANNELS * self.patch_size**2
        self._region_of = np.zeros((cfg.height, cfg.width), dtype=np.int64)
        for r in range(self.n_regions):
            y0, y1, x0, x1 = self.region_bounds(r)
            self._region_of[y0:y1, x0:x1] = r
        self.rng = np.random.default_rng(0)
        self.t = 0

    # ------------------------------------------------------------------ geometry
    def region_bounds(self, region: int) -> tuple[int, int, int, int]:
        ri, ci = divmod(region, self.cfg.region_cols)
        return (ri * self.region_h, (ri + 1) * self.region_h, ci * self.region_w, (ci + 1) * self.region_w)

    def region_neighbors(self, region: int) -> list[int]:
        """Four-adjacent regions in the region grid (peer communication graph)."""
        ri, ci = divmod(region, self.cfg.region_cols)
        out = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r2, c2 = ri + dr, ci + dc
            if 0 <= r2 < self.cfg.region_rows and 0 <= c2 < self.cfg.region_cols:
                out.append(r2 * self.cfg.region_cols + c2)
        return out

    # ------------------------------------------------------------------ episode
    def reset(self, map_seed: int) -> None:
        """Generate the map deterministically from ``map_seed``."""
        cfg = self.cfg
        rng = np.random.default_rng(map_seed)
        self.rng = np.random.default_rng(map_seed + 1_000_003)
        H, W = cfg.height, cfg.width
        fuel = _smooth(rng.random((H, W)))
        fuel = (fuel - fuel.min()) / max(float(np.ptp(fuel)), 1e-9)
        rock = _smooth(rng.random((H, W)), passes=2) < 0.18
        self.fuel = np.where(rock, 0.0, 0.2 + 0.8 * fuel)
        elev = _smooth(rng.random((H, W)), passes=4)
        self.elevation = (elev - elev.min()) / max(float(np.ptp(elev)), 1e-9)
        theta = rng.uniform(0, 2 * np.pi)
        speed = rng.uniform(0.0, cfg.wind_speed_max)
        self.wind = np.array([np.sin(theta), np.cos(theta)]) * speed  # (dy, dx)

        burnable = np.argwhere(self.fuel > 0)
        n_assets = int(rng.integers(cfg.assets_min, cfg.assets_max + 1))
        n_ign = int(rng.integers(cfg.ignitions_min, cfg.ignitions_max + 1))
        picks = rng.choice(len(burnable), size=min(len(burnable), n_assets + n_ign), replace=False)
        self.assets = burnable[picks[:n_assets]]
        self.asset_value = rng.uniform(1.0, 5.0, size=len(self.assets))
        self.asset_lost = np.zeros(len(self.assets), dtype=bool)

        self.intensity = np.zeros((H, W))
        self.burn_time = np.zeros((H, W), dtype=np.int64)
        self.burnt = np.zeros((H, W), dtype=bool)
        for y, x in burnable[picks[n_assets:]]:
            self.intensity[y, x] = 1.0

        self.crews = np.zeros((self.n_regions, self.n_crews, 2), dtype=np.int64)
        for r in range(self.n_regions):
            y0, y1, x0, x1 = self.region_bounds(r)
            for c in range(self.n_crews):
                self.crews[r, c] = (rng.integers(y0, y1), rng.integers(x0, x1))
        self.request = np.zeros(self.n_regions, dtype=bool)
        self.pending_drops: list[tuple[int, int, int]] = []  # (land_time, y, x)
        self.n_burnable = int((self.fuel > 0).sum())
        self.t = 0
        self.spread = self._spread_probability()
        self.crew_unsafe = self._crew_unsafe()

    # ------------------------------------------------------------------ dynamics
    def _spread_probability(self) -> np.ndarray:
        """Probability that each unburnt cell ignites at the next step."""
        H, W = self.intensity.shape
        no_fire = np.ones((H, W))
        pad_i = np.pad(self.intensity, 1)
        pad_e = np.pad(self.elevation, 1, mode="edge")
        speed = float(np.linalg.norm(self.wind))
        wdir = self.wind / speed if speed > 1e-9 else np.zeros(2)
        for dy, dx in _NEIGHBORS:
            src = pad_i[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
            if not src.any():
                continue
            spread_dir = np.array([-dy, -dx], dtype=float)
            spread_dir /= np.linalg.norm(spread_dir)
            cos_a = float(spread_dir @ wdir) if speed > 1e-9 else 1.0
            p_wind = np.exp(0.5 * speed) * np.exp(1.0 * speed * (cos_a - 1.0))
            src_e = pad_e[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
            p_slope = np.exp(1.5 * (self.elevation - src_e))
            diag = 0.7 if dy and dx else 1.0
            p = self.cfg.base_spread * (0.5 + self.fuel) * p_wind * p_slope * src * diag
            no_fire *= 1.0 - np.clip(p, 0.0, 1.0)
        sp = 1.0 - no_fire
        sp[(self.fuel <= 0) | (self.intensity > 0) | self.burnt] = 0.0
        return sp

    def _unsafe_cells(self) -> np.ndarray:
        """Cells where a crew counts as entrapped (paper Sec. VI-A).

        Unsafe = the cell or a 4-neighbor has spread probability above the threshold, or
        the cell itself is burning. Standing next to a burning cell is not, by itself, unsafe.
        """
        hot = self.spread > self.cfg.entrapment_threshold
        pad = np.pad(hot, 1)
        H, W = hot.shape
        adj = hot.copy()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            adj |= pad[1 + dy : 1 + dy + H, 1 + dx : 1 + dx + W]
        return adj | (self.intensity > 0)

    def _crew_unsafe(self) -> np.ndarray:
        unsafe = self._unsafe_cells()
        return unsafe[self.crews[..., 0], self.crews[..., 1]]

    def plan_drops(self, allocation: np.ndarray) -> list[tuple[int, int]]:
        """Choose drop targets: the hottest burning cells of each region (assumption A3)."""
        targets: list[tuple[int, int]] = []
        for r, k in enumerate(np.asarray(allocation, dtype=np.int64)):
            if k <= 0:
                continue
            y0, y1, x0, x1 = self.region_bounds(r)
            block = self.intensity[y0:y1, x0:x1]
            burning = np.argwhere(block > 0)
            if len(burning) == 0:
                continue
            order = np.argsort(-block[burning[:, 0], burning[:, 1]])[:k]
            targets += [(int(y0 + burning[i, 0]), int(x0 + burning[i, 1])) for i in order]
        return targets

    def step(self, crew_actions: np.ndarray, allocation: np.ndarray | None) -> tuple[float, StepInfo]:
        """Advance one step. ``allocation`` (drops per region) is applied when given."""
        cfg = self.cfg
        H, W = self.intensity.shape
        crew_costs = np.zeros((self.n_regions, self.n_crews))
        prev_unsafe = self.crew_unsafe.copy()

        # 1. Drops whose landing time has come (planned drop_latency steps earlier).
        drop_events = 0
        landing = [d for d in self.pending_drops if d[0] <= self.t]
        self.pending_drops = [d for d in self.pending_drops if d[0] > self.t]
        for _, ty, tx in landing:
            ys, xs = slice(max(0, ty - 1), ty + 2), slice(max(0, tx - 1), tx + 2)
            self.intensity[ys, xs] = 0.0
            self.fuel[ys, xs] *= 0.3
            near = np.abs(self.crews - np.array([ty, tx])).max(axis=-1) <= cfg.drop_radius
            crew_costs += near
            drop_events += int(near.sum())
        if allocation is not None:
            land = self.t + cfg.drop_latency
            self.pending_drops += [(land, y, x) for y, x in self.plan_drops(allocation)]

        # 2. Crew actions.
        self.request[:] = False
        acts = np.asarray(crew_actions, dtype=np.int64).reshape(self.n_regions, self.n_crews)
        for r in range(self.n_regions):
            y0, y1, x0, x1 = self.region_bounds(r)
            for c in range(self.n_crews):
                a = int(acts[r, c])
                y, x = self.crews[r, c]
                if a in MOVES:
                    dy, dx = MOVES[a]
                    self.crews[r, c] = (min(max(y + dy, y0), y1 - 1), min(max(x + dx, x0), x1 - 1))
                elif a == 5:
                    for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
                        yy, xx = y + dy, x + dx
                        if 0 <= yy < H and 0 <= xx < W and self.intensity[yy, xx] > 0:
                            self.intensity[yy, xx] -= cfg.suppression_amount
                            if self.intensity[yy, xx] <= 0:
                                self.intensity[yy, xx] = 0.0
                                self.fuel[yy, xx] *= 0.5
                elif a == 6 and self.intensity[y, x] <= 0:
                    self.fuel[y, x] = 0.0
                elif a == 7:
                    self.request[r] = True

        # 3. Fire spread and burn-out.
        sp = self._spread_probability()
        ignite = self.rng.random((H, W)) < sp
        newly = int(ignite.sum())
        burning = self.intensity > 0
        self.intensity[burning] = np.minimum(1.0, self.intensity[burning] + 0.15)
        self.burn_time[burning] += 1
        done_burning = burning & (self.burn_time >= cfg.burn_duration)
        self.burnt |= done_burning
        self.intensity[done_burning] = 0.0
        self.fuel[done_burning] = 0.0
        self.intensity[ignite] = 0.6

        # 4. Asset losses.
        lost_now = (self.intensity[self.assets[:, 0], self.assets[:, 1]] > 0) & ~self.asset_lost
        value_lost = float(self.asset_value[lost_now].sum())
        self.asset_lost |= lost_now

        # 5. Observation-time quantities and entrapment onset (assumption A4).
        self.spread = self._spread_probability()
        self.crew_unsafe = self._crew_unsafe()
        onset = self.crew_unsafe & ~prev_unsafe
        crew_costs += onset
        self.t += 1

        total_value = float(self.asset_value.sum()) or 1.0
        reward = -10.0 * newly / max(self.n_burnable, 1) - 10.0 * value_lost / total_value
        return reward, StepInfo(
            crew_costs=crew_costs,
            entrapment_events=int(onset.sum()),
            drop_events=drop_events,
            newly_ignited=newly,
            asset_value_lost=value_lost,
        )

    @property
    def done(self) -> bool:
        return self.t >= self.cfg.episode_length

    # ------------------------------------------------------------------ observations
    def reports(self) -> np.ndarray:
        """True 16-dimensional regional reports m_t^i (assumption A5 lists the fields)."""
        out = np.zeros((self.n_regions, REPORT_DIM), dtype=np.float32)
        threatened = self.spread > 0.3
        for r in range(self.n_regions):
            y0, y1, x0, x1 = self.region_bounds(r)
            block = self.intensity[y0:y1, x0:x1]
            burning = np.argwhere(block > 0)
            area = block.size
            out[r, 0] = len(burning) / area
            out[r, 1] = block[block > 0].mean() if len(burning) else 0.0
            out[r, 2] = self.spread[y0:y1, x0:x1].max()
            if len(burning):
                out[r, 3] = burning[:, 0].mean() / self.region_h
                out[r, 4] = burning[:, 1].mean() / self.region_w
            else:
                out[r, 3:5] = 0.5
            in_r = self._region_of[self.assets[:, 0], self.assets[:, 1]] == r
            if in_r.any():
                ay, ax = self.assets[in_r, 0], self.assets[in_r, 1]
                alive = ~self.asset_lost[in_r]
                out[r, 5] = float((threatened[ay, ax] & alive).mean())
                vals = self.asset_value[in_r]
                out[r, 6] = float(vals[alive].sum() / vals.sum())
            else:
                out[r, 6] = 1.0
            for c in range(min(self.n_crews, 3)):
                out[r, 7 + 2 * c] = (self.crews[r, c, 0] - y0) / self.region_h
                out[r, 8 + 2 * c] = (self.crews[r, c, 1] - x0) / self.region_w
            out[r, 13] = float(self.request[r])
            out[r, 14:16] = self.wind
        return out

    def border_view(self, observer: int, target: int) -> np.ndarray | None:
        """What region ``observer`` sees of adjacent region ``target`` (verification replies).

        Returns ``[burning cells, max spread probability]`` over the band of ``target`` that lies
        within ``halo_width`` cells of the shared border, or None if the regions are not adjacent.
        This is the only information a neighbor has about another region; no agent sees the
        target's full state.
        """
        if target not in self.region_neighbors(observer):
            return None
        y0, y1, x0, x1 = self.region_bounds(target)
        oy0, oy1, ox0, ox1 = self.region_bounds(observer)
        w = self.cfg.halo_width
        if oy1 == y0:  # observer above target
            y1 = min(y1, y0 + w)
        elif oy0 == y1:  # observer below target
            y0 = max(y0, y1 - w)
        elif ox1 == x0:  # observer left of target
            x1 = min(x1, x0 + w)
        else:  # observer right of target
            x0 = max(x0, x1 - w)
        band_i = self.intensity[y0:y1, x0:x1]
        band_s = self.spread[y0:y1, x0:x1]
        return np.array([float((band_i > 0).sum()), float(band_s.max())], dtype=np.float32)

    def crew_patches(self) -> np.ndarray:
        """Local (2R+1)^2 patches of intensity, spread probability, and fuel: [N, C, 3, P, P]."""
        R = self.cfg.patch_radius
        stack = np.stack([self.intensity, self.spread, self.fuel])
        padded = np.pad(stack, ((0, 0), (R, R), (R, R)))
        out = np.zeros(
            (self.n_regions, self.n_crews, PATCH_CHANNELS, self.patch_size, self.patch_size), dtype=np.float32
        )
        for r in range(self.n_regions):
            for c in range(self.n_crews):
                y, x = self.crews[r, c]
                out[r, c] = padded[:, y : y + 2 * R + 1, x : x + 2 * R + 1]
        return out

    def crew_drop_offsets(self, targets: list[tuple[int, int]]) -> np.ndarray:
        """For each crew, a [P, P] mask of drop targets visible in its patch."""
        R = self.cfg.patch_radius
        out = np.zeros((self.n_regions, self.n_crews, self.patch_size, self.patch_size), dtype=bool)
        for ty, tx in targets:
            dy = ty - self.crews[..., 0] + R
            dx = tx - self.crews[..., 1] + R
            ok = (dy >= 0) & (dy < self.patch_size) & (dx >= 0) & (dx < self.patch_size)
            for r, c in np.argwhere(ok):
                out[r, c, dy[r, c], dx[r, c]] = True
        return out

    def global_state(self) -> np.ndarray:
        """Centralized-critic input: all true reports plus normalized time."""
        return np.concatenate([self.reports().ravel(), [self.t / self.cfg.episode_length]]).astype(np.float32)

    # ------------------------------------------------------------------ episode metrics
    def asset_protection_rate(self) -> float:
        """APR: percentage of total asset value not burned (paper Sec. VI-E)."""
        total = float(self.asset_value.sum())
        return 100.0 * (1.0 - float(self.asset_value[self.asset_lost].sum()) / total) if total else 100.0

    def burned_area(self) -> float:
        """BA: percentage of burnable cells burnt or burning."""
        affected = int((self.burnt | (self.intensity > 0)).sum())
        return 100.0 * affected / max(self.n_burnable, 1)
