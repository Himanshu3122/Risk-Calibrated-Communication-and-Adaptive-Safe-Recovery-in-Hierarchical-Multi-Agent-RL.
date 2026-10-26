"""RiCoR agent system: decentralized execution of paper Algorithm 1 over a batch of episodes.

One :class:`RiCoRSystem` owns every network. :meth:`RiCoRSystem.run_episodes` steps a batch
of HierFire episodes in lockstep, each with its own fault-prone :class:`~ricor.comm.Channel`,
and returns both training data and evaluation records.

Method switches in :class:`~ricor.config.MethodConfig` turn the same code into the
baselines (H-MAPPO, H-MAPPO+Drop, ADMAC-H, HPSF) and the Table VII ablations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from torch import nn

from ricor.comm.channel import AGE_CAP, PEER, UPLINK, Channel
from ricor.comm.faults import FaultSpec
from ricor.config import ExperimentConfig
from ricor.core.gate import categorical_kl, decision_sensitivity, gate_weights
from ricor.core.modes import N_MODES, FixedRule, Mode, admissible, mode_reward, mode_summary
from ricor.core.reliability import beta_from_evidence, beta_mean_var, reliability
from ricor.core.shield import shield
from ricor.envs.hierfire import N_CREW_ACTIONS, REPORT_DIM, HierFireEnv
from ricor.models import (
    META_DIM,
    ChannelEstimator,
    CoordinatorPolicy,
    CredibilityNet,
    MissingInfoPredictor,
    ModeSelector,
    QuantileCritic,
    RegionalPolicy,
    ValueCritic,
)

VERIFY_SPREAD_TOL = 0.05  # assumption A12: slack on the max-spread lower bound
VERIFY_CELL_TOL = 0.5  # assumption A12: slack (cells) on the burning-cell lower bound
THREAT = [0, 1, 2, 5]
WIND = [14, 15]


@dataclass
class EpisodeSlot:
    """Mutable per-episode state outside the simulator."""

    env: HierFireEnv
    channel: Channel
    mode: Mode = Mode.NORMAL
    since_switch: int = 0
    outage: int = 0
    rule: FixedRule | None = None
    last_alloc: np.ndarray = field(default_factory=lambda: np.zeros(0))
    last_decision: int = -1
    decision_targets: dict[int, list[tuple[int, int]]] = field(default_factory=dict)
    sent_decision: dict[int, int] = field(default_factory=dict)
    fallback_goal: np.ndarray | None = None
    fallback_seq: int = -1
    prev_content: np.ndarray | None = None
    trusted: np.ndarray | None = None
    trusted_valid: np.ndarray | None = None
    rep_pos: np.ndarray | None = None  # Beta-reputation evidence per content link
    rep_neg: np.ndarray | None = None


@dataclass
class Rollout:
    """Stacked rollout tensors ([T, B, ...]) and evaluation records."""

    data: dict[str, torch.Tensor]
    episode_metrics: list[dict[str, Any]]
    reliability_records: dict[str, np.ndarray]
    step_rewards: np.ndarray  # [T, B]


class RiCoRSystem(nn.Module):
    """All networks plus the execution logic of Algorithm 1."""

    def __init__(self, cfg: ExperimentConfig) -> None:
        super().__init__()
        self.cfg = cfg
        e, m, mc = cfg.env, cfg.model, cfg.method
        probe = HierFireEnv(e)
        self.N, self.C = probe.n_regions, probe.n_crews
        self.P = probe.patch_size
        self.patch_dim = probe.patch_dim
        self.peers = [probe.region_neighbors(r) for r in range(self.N)]
        self.M = max(len(p) for p in self.peers)
        probe_channel = Channel(self.N, self.peers, REPORT_DIM, self.N)
        self.L = probe_channel.n_links
        self.content_idx = np.array([i for i, link in enumerate(probe_channel.links) if link[2] in (UPLINK, PEER)])
        self.uplink_idx = probe_channel.uplink_idx
        self.downlink_idx = probe_channel.downlink_idx
        self.peer_idx = probe_channel.peer_idx
        self.link_src = np.array([link[0] for link in probe_channel.links])
        self.link_dst = np.array([link[1] for link in probe_channel.links])

        self.goal_dim = self.N + 1
        self.obs_dim = (
            REPORT_DIM + self.C * self.patch_dim + self.C * self.P * self.P + self.goal_dim + N_MODES + self.N
        )
        self.coord_extra = N_MODES + self.N + 1
        self.coord_local_dim = self.N * REPORT_DIM + 2 * self.N + self.coord_extra
        self.global_dim = self.N * REPORT_DIM + 1
        # Receivers of each content link, for per-receiver consistency features.
        ci = self.content_idx
        self.recv_groups = {int(n): np.flatnonzero(self.link_dst[ci] == n) for n in np.unique(self.link_dst[ci])}
        self.crew_feat_dim = self.patch_dim + self.P * self.P + REPORT_DIM
        self.cred_dim = 4 * REPORT_DIM + (0 if mc.split_reliability else META_DIM)

        self.regional = RegionalPolicy(self.obs_dim, self.C, m.hidden, m.message_dim, m.attention_heads)
        self.coordinator = CoordinatorPolicy(self.coord_extra, m.hidden, m.message_dim)
        self.value = ValueCritic(self.global_dim, m.hidden)
        self.cost_critic = QuantileCritic(self.global_dim, m.hidden, m.n_quantiles)
        # Decentralized coordinator cost critic: input is only what the coordinator received
        # (gated reports, gate weights, validity) plus its own mode, allocation, and clock.
        self.coord_cost = QuantileCritic(self.coord_local_dim, m.hidden, m.n_quantiles)
        self.crew_q = QuantileCritic(self.crew_feat_dim, m.hidden, m.n_quantiles, n_actions=N_CREW_ACTIONS)
        self.channel_est = ChannelEstimator(m.gru_hidden)
        self.credibility = CredibilityNet(self.cred_dim, m.credibility_hidden)
        self.mode_selector = ModeSelector(m.hidden)
        self.predictor = MissingInfoPredictor(m.gru_hidden)
        self.register_buffer("temp_ch", torch.tensor(1.0))
        self.register_buffer("temp_cr", torch.tensor(1.0))
        self.register_buffer("lagrange", torch.tensor(0.0))

    # ------------------------------------------------------------------ helpers
    @property
    def device(self) -> torch.device:
        return self.temp_ch.device

    def _t(self, x: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(x), dtype=dtype, device=self.device)

    def cred_features(
        self, content: torch.Tensor, prev: torch.Tensor, median: torch.Tensor, recv: torch.Tensor, meta: torch.Tensor
    ) -> torch.Tensor:
        """Credibility input: message, temporal change, deviation from other senders, receiver's view."""
        parts = [content, content - prev, content - median, recv]
        if not self.cfg.method.split_reliability:
            parts.append(meta)
        return torch.cat(parts, -1)

    def coord_local(
        self, slots: torch.Tensor, w: torch.Tensor, valid: torch.Tensor, extra: torch.Tensor
    ) -> torch.Tensor:
        """Coordinator-local critic input: gated received reports, weights, validity, own state."""
        vf = valid.float()
        gated = (slots * (w * vf).unsqueeze(-1)).flatten(1)
        return torch.cat([gated, w, vf, extra], -1)

    def reliability_terms(self, p_logit: torch.Tensor, cred_x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Eqs. (4)-(6) with temperature scaling applied to both factors."""
        mc = self.cfg.method
        p_ch = torch.sigmoid(p_logit / self.temp_ch)
        e_pos, e_neg = self.credibility(cred_x)
        alpha, beta = beta_from_evidence(e_pos, e_neg)
        r_raw, var = beta_mean_var(alpha, beta)
        r_bar = torch.sigmoid(torch.logit(r_raw.clamp(1e-5, 1 - 1e-5)) / self.temp_cr)
        sigma = var.sqrt()
        rho_bar, rho_lower = reliability(p_ch, r_bar, sigma, mc.lambda_u, mc.split_reliability, mc.uncertainty)
        return {"p_ch": p_ch, "r_bar": r_bar, "sigma": sigma, "rho_bar": rho_bar, "rho_lower": rho_lower}

    # ------------------------------------------------------------------ episodes
    def new_slot(
        self, map_seed: int, fault: FaultSpec, channel_seed: int, window: tuple[int, int] | None
    ) -> EpisodeSlot:
        env = HierFireEnv(self.cfg.env)
        env.reset(map_seed)
        channel = Channel(self.N, self.peers, REPORT_DIM, self.N)
        channel.reset(fault, seed=channel_seed, window=window)
        mc = self.cfg.method
        rule = FixedRule(mc.fixed_tau, mc.fixed_psi_v, mc.fixed_sigma_v, mc.fixed_outage, mc.hysteresis)
        return EpisodeSlot(
            env=env,
            channel=channel,
            rule=rule,
            last_alloc=np.zeros(self.N),
            prev_content=np.zeros((self.L, REPORT_DIM), np.float32),
            trusted=np.zeros((self.L, REPORT_DIM), np.float32),
            trusted_valid=np.zeros(self.L, bool),
            rep_pos=np.zeros(len(self.content_idx)),
            rep_neg=np.zeros(len(self.content_idx)),
        )

    @torch.no_grad()
    def run_episodes(
        self,
        map_seeds: list[int],
        faults: list[FaultSpec],
        channel_seeds: list[int],
        stochastic: bool = True,
        training: bool = False,
        windows: list[tuple[int, int] | None] | None = None,
        record_reliability: bool = False,
    ) -> Rollout:
        """Run ``len(map_seeds)`` full episodes in lockstep and collect data."""
        cfg, mc = self.cfg, self.cfg.method
        B, N, C, L = len(map_seeds), self.N, self.C, self.L
        windows = windows or [None] * B
        slots = [
            self.new_slot(s, f, cs, w) for s, f, cs, w in zip(map_seeds, faults, channel_seeds, windows, strict=True)
        ]
        T = cfg.env.episode_length
        K = cfg.env.coordinator_period
        H_ch = torch.zeros(B * L, self.channel_est.hidden_size, device=self.device)
        Lc = len(self.content_idx)
        H_pr = torch.zeros(B * Lc, self.predictor.hidden_size, device=self.device)
        lam = float(self.lagrange)
        gen = torch.Generator(device="cpu").manual_seed(int(channel_seeds[0]) + 7)

        buf: dict[str, list[Any]] = {
            k: []
            for k in (
                "obs_r",
                "msgs_r",
                "w_r",
                "valid_r",
                "act_r",
                "logp_r",
                "slots_c",
                "w_c",
                "valid_c",
                "extra_c",
                "drops_c",
                "logp_c",
                "decided_c",
                "mode_x",
                "mode_mask",
                "mode_a",
                "mode_logp",
                "mode_v",
                "mode_decided",
                "gstate",
                "coord_x",
                "reward",
                "cost",
                "mode_reward",
                "crew_feat",
                "crew_exec",
                "crew_cost",
                "meta",
                "cred_x",
                "y_ch",
                "y_cr",
                "has_c",
                "pred_held",
                "pred_age",
                "pred_target",
            )
        }
        rel_rec: dict[str, list[np.ndarray]] = {k: [] for k in ("rho_bar", "p_ch", "r_bar", "y_ch", "y_cr", "w")}
        step_rewards = np.zeros((T, B))
        ep = [
            dict(
                bytes=0,
                queries=0,
                shield=0,
                mode_counts=np.zeros(N_MODES),
                latency=0.0,
                cost=0.0,
                entrapment=0,
                drops_near=0,
            )
            for _ in range(B)
        ]

        for t in range(T):
            reports = np.stack([s.env.reports() for s in slots])  # [B, N, 16]
            patches = np.stack([s.env.crew_patches() for s in slots])  # [B, N, C, 3, P, P]
            gstate = np.stack([s.env.global_state() for s in slots])
            goals = np.stack([np.tile(s.last_alloc / max(cfg.env.n_drops, 1), (N, 1)) for s in slots])
            deliveries = []
            for b, s in enumerate(slots):
                s.sent_decision[t] = s.last_decision
                d = s.channel.transmit(t, reports[b], goals[b])
                deliveries.append(d)
                ep[b]["bytes"] += d.bytes_sent

            meta = np.stack([np.stack([d.arrived, d.age / AGE_CAP, d.gap], -1) for d in deliveries]).astype(np.float32)
            content = np.stack([d.content[:, :REPORT_DIM] for d in deliveries])  # [B, L, 16]
            has = np.stack([d.has_content for d in deliveries])  # [B, L]
            y_ch = np.stack([d.y_ch for d in deliveries])
            y_cr = np.stack([d.y_cr for d in deliveries])
            age = np.stack([d.age for d in deliveries])
            ci = self.content_idx
            if mc.no_comm:  # lower bound: nothing that arrives over the network is used
                has = np.zeros_like(has)
            t_dec = time.perf_counter()  # decision latency excludes simulation and channel time

            # ---- reliability (Eqs. 4-6)
            meta_t = self._t(meta)
            p_logit, H_ch = self.channel_est.step(meta_t.view(B * L, META_DIM), H_ch)
            p_logit = p_logit.view(B, L)
            prev = np.stack([s.prev_content[ci] for s in slots])
            median = self._receiver_median(content[:, ci], has[:, ci])
            recv = np.zeros((B, Lc, REPORT_DIM), np.float32)
            dst = self.link_dst[ci]
            for j, dnode in enumerate(dst):
                if dnode > 0:  # the receiving region's own observation; the coordinator has none
                    recv[:, j] = reports[:, dnode - 1]
            cred_x = self.cred_features(
                self._t(content[:, ci]), self._t(prev), self._t(median), self._t(recv), meta_t[:, ci]
            )
            rel = self.reliability_terms(p_logit[:, ci], cred_x)
            if mc.reputation:
                rep = self._reputation(slots, deliveries, content[:, ci], prev, median, recv, dst)
                ones = torch.ones_like(rep)
                rel = {"p_ch": ones, "r_bar": rep, "sigma": torch.zeros_like(rep), "rho_bar": rep, "rho_lower": rep}
            has_c = self._t(has[:, ci], torch.bool)
            drop_masks = [self._drop_masks(s, d) for s, d in zip(slots, deliveries, strict=True)]

            # ---- message content used by receivers (HPSF prediction, dropout augmentation)
            msg = self._t(content[:, ci])
            if mc.predict_missing:
                pred, H_pr = self.predictor.step(msg.view(B * Lc, -1), self._t(age[:, ci] / AGE_CAP).view(-1), H_pr)
                stale = self._t((age[:, ci] > 1) & has[:, ci], torch.bool)
                msg = torch.where(stale.unsqueeze(-1), pred.view(B, Lc, -1), msg)
                has_c = has_c | stale
            valid = has_c.clone()
            if training and mc.message_dropout > 0:
                keep = torch.rand(B, Lc, generator=gen) >= mc.message_dropout
                valid &= keep.to(self.device)
            if training and mc.message_noise > 0:
                msg = msg + mc.message_noise * torch.randn(msg.shape, generator=gen).to(self.device)

            # ---- risk and gate (Eqs. 7-8)
            regional_obs = self._regional_obs(slots, reports, patches, deliveries, t, drop_masks)
            crew_feat = self._crew_features(slots, reports, patches, drop_masks)
            extra_c = np.zeros((B, self.coord_extra), np.float32)
            for b, s in enumerate(slots):
                extra_c[b, int(s.mode)] = 1.0
                extra_c[b, N_MODES : N_MODES + N] = s.last_alloc / max(cfg.env.n_drops, 1)
                extra_c[b, -1] = t / T
            decide = t % K == 0
            if mc.oracle_gate:  # privileged reference: accept exactly the fresh, unaltered messages
                label = self._t(y_ch[:, ci] * y_cr[:, ci])
                psi = torch.zeros(B, Lc, device=self.device)
                w = torch.where(valid, label, torch.zeros_like(label))
            elif mc.gating:
                # Reference weights w0: the same gate with kappa = 0. They depend only on rho_lower,
                # so psi (Eq. 7) is computed before, and independently of, the final gate (Eq. 8).
                w_base, _ = gate_weights(
                    rel["rho_lower"], torch.zeros_like(rel["rho_lower"]), mc.tau0, 0.0, mc.gate_temperature
                )
                w_base = torch.where(valid, w_base, torch.zeros_like(w_base))
                psi = torch.zeros(B, Lc, device=self.device)
                if mc.kappa > 0:
                    psi = self._sensitivity(regional_obs, crew_feat, extra_c, decide, msg, w_base, valid)
                w, _ = gate_weights(rel["rho_lower"], psi, mc.tau0, mc.kappa, mc.gate_temperature)
                w = torch.where(valid, w, torch.zeros_like(w))
            else:
                psi = torch.zeros(B, Lc, device=self.device)
                w = valid.float()

            if record_reliability:
                hc = has[:, ci]
                for key, val in (("rho_bar", rel["rho_bar"]), ("p_ch", rel["p_ch"]), ("r_bar", rel["r_bar"]), ("w", w)):
                    rel_rec[key].append(val.cpu().numpy()[hc])
                rel_rec["y_ch"].append(y_ch[:, ci][hc])
                rel_rec["y_cr"].append(y_cr[:, ci][hc])

            # ---- operating mode
            w_np, msg_np = w.cpu().numpy(), msg.cpu().numpy()
            up_pos = np.searchsorted(ci, self.uplink_idx)
            mode_x = np.zeros((B, 6 + N_MODES), np.float32)
            mode_mask = np.ones((B, N_MODES), bool)
            mode_a = np.zeros(B, np.int64)
            mode_logp = np.zeros(B, np.float32)
            mode_v = np.zeros(B, np.float32)
            for b, s in enumerate(slots):
                s.outage = 0 if deliveries[b].arrived[self.uplink_idx].any() else s.outage + 1
                rl = rel["rho_lower"][b, up_pos].cpu().numpy()
                xi = mode_summary(
                    rl,
                    psi[b, up_pos].cpu().numpy(),
                    rel["sigma"][b, up_pos].cpu().numpy(),
                    s.outage,
                    s.since_switch,
                    mc.hysteresis,
                )
                mode_x[b, :6] = xi
                mode_x[b, 6 + int(s.mode)] = 1.0
                mode_mask[b] = admissible(s.mode, s.since_switch, mc.hysteresis, mc.verify_mode)
                if mc.modes == "fixed":
                    mode_a[b] = int(s.rule.choose(s.mode, xi, s.outage, s.since_switch, mc.verify_mode))
                elif mc.modes == "none":
                    mode_a[b] = int(Mode.NORMAL)
            if mc.modes == "learned":
                logits, v = self.mode_selector(self._t(mode_x), self._t(mode_mask, torch.bool))
                dist = torch.distributions.Categorical(logits=logits)
                a = dist.sample() if stochastic else logits.argmax(-1)
                mode_a, mode_logp, mode_v = a.cpu().numpy(), dist.log_prob(a).cpu().numpy(), v.cpu().numpy()

            # ---- apply mode effects and verification
            n_queries = np.zeros(B, np.int64)
            for b, s in enumerate(slots):
                new_mode = Mode(int(mode_a[b]))
                switched = new_mode != s.mode
                if switched and new_mode == Mode.FALLBACK:
                    s.fallback_goal = deliveries[b].content[self.downlink_idx, : self.N].copy()
                    s.fallback_seq = t
                s.since_switch = 0 if switched else s.since_switch + 1
                s.mode = new_mode
                ep[b]["mode_counts"][int(new_mode)] += 1
                n_queries[b] = self._apply_mode(b, s, w_np, msg_np, valid, t, up_pos)
                ep[b]["queries"] += int(n_queries[b])
                ep[b]["bytes"] += int(n_queries[b]) * (16 + 4 * REPORT_DIM)
                s.prev_content[:] = content[b]
            w_eff = self._t(w_np)
            msg_eff = self._t(msg_np)
            valid_eff = valid & (w_eff > 0)

            # ---- coordinator (every K steps, held in fallback)
            slots_c = msg_eff[:, up_pos]
            w_c, valid_c = w_eff[:, up_pos], valid_eff[:, up_pos]
            for b, s in enumerate(slots):  # the decision uses the mode chosen at this step
                extra_c[b, :N_MODES] = 0.0
                extra_c[b, int(s.mode)] = 1.0
            drops = np.zeros((B, cfg.env.n_drops), np.int64)
            logp_c = np.zeros(B, np.float32)
            allocations: list[np.ndarray | None] = [None] * B
            if decide:
                logits_c = self.coordinator(slots_c, w_c, valid_c, self._t(extra_c))
                dist_c = torch.distributions.Categorical(logits=logits_c.unsqueeze(1).expand(B, cfg.env.n_drops, N))
                if stochastic:
                    d_a = dist_c.sample()
                else:
                    d_a = self._self_greedy_drops(torch.softmax(logits_c, -1), cfg.env.n_drops)
                drops, logp_c = d_a.cpu().numpy(), dist_c.log_prob(d_a).sum(-1).cpu().numpy()
                for b, s in enumerate(slots):
                    if s.mode == Mode.FALLBACK and s.last_decision >= 0:
                        allocations[b] = s.last_alloc.copy()
                    else:
                        allocations[b] = np.bincount(drops[b], minlength=N).astype(np.float64)
            decided_c = np.array([decide and s.mode != Mode.FALLBACK for s in slots])

            # ---- regional policies and shield
            msgs_r, w_r, valid_r = self._regional_messages(msg_eff, w_eff, valid_eff)
            logits_r = self.regional(
                self._t(regional_obs).view(B * N, -1),
                msgs_r.view(B * N, self.M, -1),
                w_r.view(B * N, self.M),
                valid_r.view(B * N, self.M),
            )
            dist_r = torch.distributions.Categorical(logits=logits_r)
            act = dist_r.sample() if stochastic else logits_r.argmax(-1)
            logp_r = dist_r.log_prob(act).sum(-1).view(B, N).cpu().numpy()
            act_np = act.view(B, N, C).cpu().numpy()
            probs = torch.softmax(logits_r, -1).view(B, N, C, -1).cpu().numpy()
            executed = act_np.copy()
            if mc.shield:
                for b, s in enumerate(slots):
                    masks = drop_masks[b]
                    for r in range(N):
                        for c in range(C):
                            a_exec, hit = shield(
                                int(act_np[b, r, c]),
                                probs[b, r, c],
                                patches[b, r, c],
                                masks[r, c],
                                cfg.env.entrapment_threshold,
                                cfg.env.drop_radius,
                            )
                            executed[b, r, c] = a_exec
                            ep[b]["shield"] += int(hit)

            dec_seconds = (time.perf_counter() - t_dec) / B

            # Coordinator-local critic input with the allocation in force after this step.
            extra_now = extra_c.copy()
            for b, s in enumerate(slots):
                alloc_now = allocations[b] if allocations[b] is not None else s.last_alloc
                extra_now[b, N_MODES : N_MODES + N] = alloc_now / max(cfg.env.n_drops, 1)
            coord_x = self.coord_local(slots_c, w_c, valid_c, self._t(extra_now))

            # ---- environment step
            rewards = np.zeros(B, np.float32)
            costs = np.zeros(B, np.float32)
            crew_cost = np.zeros((B, N, C), np.float32)
            m_rew = np.zeros(B, np.float32)
            for b, s in enumerate(slots):
                alloc = allocations[b]
                r_t, info = s.env.step(executed[b], alloc)
                if alloc is not None:
                    s.last_alloc = alloc
                    s.last_decision = t
                    land = t + cfg.env.drop_latency
                    s.decision_targets[t] = [(y, x) for (lt, y, x) in s.env.pending_drops if lt == land]
                rewards[b], costs[b] = r_t, info.crew_costs.sum()
                crew_cost[b] = info.crew_costs
                m_rew[b] = mode_reward(
                    r_t,
                    bool(slots[b].since_switch == 0 and t > 0),
                    int(n_queries[b]),
                    costs[b],
                    lam,
                    mc.switch_cost,
                    mc.query_cost,
                )
                ep[b]["cost"] += float(costs[b])
                ep[b]["entrapment"] += info.entrapment_events
                ep[b]["drops_near"] += info.drop_events
                ep[b]["latency"] += dec_seconds
            step_rewards[t] = rewards

            if training:
                true_sender = np.stack([reports[b][self.link_src[ci] - 1] for b in range(B)])
                for key, val in (
                    ("obs_r", regional_obs),
                    ("msgs_r", msgs_r.cpu().numpy()),
                    ("w_r", w_r.cpu().numpy()),
                    ("valid_r", valid_r.cpu().numpy()),
                    ("act_r", act_np),
                    ("logp_r", logp_r),
                    ("slots_c", slots_c.cpu().numpy()),
                    ("w_c", w_c.cpu().numpy()),
                    ("valid_c", valid_c.cpu().numpy()),
                    ("extra_c", extra_c),
                    ("drops_c", drops),
                    ("logp_c", logp_c),
                    ("decided_c", decided_c),
                    ("mode_x", mode_x),
                    ("mode_mask", mode_mask),
                    ("mode_a", mode_a),
                    ("mode_logp", mode_logp),
                    ("mode_v", mode_v),
                    ("mode_decided", np.full(B, mc.modes == "learned")),
                    ("gstate", gstate),
                    ("coord_x", coord_x.cpu().numpy()),
                    ("reward", rewards),
                    ("cost", costs),
                    ("mode_reward", m_rew),
                    ("crew_feat", crew_feat),
                    ("crew_exec", executed),
                    ("crew_cost", crew_cost),
                    ("meta", meta),
                    ("cred_x", cred_x.cpu().numpy()),
                    ("y_ch", y_ch),
                    ("y_cr", y_cr[:, ci]),
                    ("has_c", has[:, ci]),
                    ("pred_held", content[:, ci]),
                    ("pred_age", age[:, ci] / AGE_CAP),
                    ("pred_target", true_sender),
                ):
                    buf[key].append(np.asarray(val))

        metrics = []
        for b, s in enumerate(slots):
            metrics.append(
                {
                    "apr": s.env.asset_protection_rate(),
                    "ba": s.env.burned_area(),
                    "sv": ep[b]["cost"],
                    "entrapment": ep[b]["entrapment"],
                    "drops_near_crew": ep[b]["drops_near"],
                    "return": float(step_rewards[:, b].sum()),
                    "comm_kb": ep[b]["bytes"] / 1024.0,
                    "queries": ep[b]["queries"],
                    "shield_interventions": ep[b]["shield"],
                    "mode_occupancy": (ep[b]["mode_counts"] / T).tolist(),
                    "latency_ms": 1000.0 * ep[b]["latency"] / T,
                }
            )
        data = {k: torch.from_numpy(np.ascontiguousarray(np.stack(v))) for k, v in buf.items() if v} if training else {}
        rec = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in rel_rec.items()}
        return Rollout(data=data, episode_metrics=metrics, reliability_records=rec, step_rewards=step_rewards)

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _self_greedy_drops(probs: torch.Tensor, n_drops: int) -> torch.Tensor:
        """Deterministic allocation: largest-remainder rounding of n_drops * probs (assumption A13)."""
        raw = probs * n_drops
        counts = raw.floor().long()
        remainder = raw - counts
        missing = n_drops - counts.sum(-1)
        for b in range(probs.shape[0]):
            if missing[b] > 0:
                counts[b, remainder[b].topk(int(missing[b])).indices] += 1
        return torch.stack([torch.repeat_interleave(torch.arange(probs.shape[1]), c) for c in counts.cpu()]).to(
            probs.device
        )

    def _known_targets(self, s: EpisodeSlot, d: Any, region: int) -> list[tuple[int, int]]:
        """Drop targets region ``region`` learns from its own newest goal message (assumption A3)."""
        seq = int(d.seq[self.downlink_idx[region]])
        if seq < 0 or self.cfg.method.no_comm:
            return []
        return sorted(s.decision_targets.get(s.sent_decision.get(seq, -1), []))

    def _drop_masks(self, s: EpisodeSlot, d: Any) -> np.ndarray:
        """[N, C, P, P] drop-target masks; row r uses only what region r was told."""
        out = np.zeros((self.N, self.C, self.P, self.P), dtype=bool)
        for r in range(self.N):
            targets = self._known_targets(s, d, r)
            if targets:
                out[r] = s.env.crew_drop_offsets(targets)[r]
        return out

    def _receiver_median(self, content: np.ndarray, has: np.ndarray) -> np.ndarray:
        """Per content link, the median of the messages its receiver currently holds: [B, Lc, D]."""
        B, Lc, D = content.shape
        out = np.zeros((B, Lc, D), np.float32)
        for pos in self.recv_groups.values():
            vals, h = content[:, pos], has[:, pos]
            for b in range(B):
                if h[b].any():
                    out[b, pos] = np.median(vals[b][h[b]], 0)
        return out

    def _reputation(
        self,
        slots: list[EpisodeSlot],
        deliveries: list[Any],
        content: np.ndarray,
        prev: np.ndarray,
        median: np.ndarray,
        recv: np.ndarray,
        dst: np.ndarray,
    ) -> torch.Tensor:
        """Beta-reputation baseline: expected trust (1 + pos) / (2 + pos + neg) per content link.

        Each newly arrived message is an interaction whose outcome is judged from observable
        consistency only: its wind fields (global in HierFire) must match the receiver's own
        wind (or, for the coordinator, the median received wind), and its threat fields must not
        jump by more than ``reputation_tol`` from the link's previous message. Evidence decays by
        ``reputation_decay`` per arrival, in the style of the Beta Reputation System and TRAVOS.
        """
        mc = self.cfg.method
        ci = self.content_idx
        out = np.zeros(content.shape[:2], np.float32)
        for b, s in enumerate(slots):
            arrived = deliveries[b].arrived[ci]
            wind_ref = np.where((dst > 0)[:, None], recv[b][:, WIND], median[b][:, WIND])
            wind_ok = np.abs(content[b][:, WIND] - wind_ref).max(-1) <= mc.reputation_wind_tol
            first = np.abs(prev[b]).sum(-1) == 0  # no earlier message on this link
            steady = np.abs(content[b][:, THREAT] - prev[b][:, THREAT]).max(-1) <= mc.reputation_tol
            ok = wind_ok & (steady | first)
            s.rep_pos[arrived] = mc.reputation_decay * s.rep_pos[arrived] + ok[arrived]
            s.rep_neg[arrived] = mc.reputation_decay * s.rep_neg[arrived] + (~ok)[arrived]
            out[b] = (1.0 + s.rep_pos) / (2.0 + s.rep_pos + s.rep_neg)
        return self._t(out)

    def _regional_obs(
        self,
        slots: list[EpisodeSlot],
        reports: np.ndarray,
        patches: np.ndarray,
        deliveries: list[Any],
        t: int,
        drop_masks: list[np.ndarray],
    ) -> np.ndarray:
        B, N = len(slots), self.N
        out = np.zeros((B, N, self.obs_dim), np.float32)
        for b, s in enumerate(slots):
            masks = drop_masks[b].astype(np.float32)
            d = deliveries[b]
            for r in range(N):
                li = self.downlink_idx[r]
                if self.cfg.method.no_comm:
                    goal, g_age = np.zeros(N, np.float32), AGE_CAP
                elif s.mode == Mode.FALLBACK and s.fallback_goal is not None:
                    goal, g_age = s.fallback_goal[r], t - s.fallback_seq
                else:
                    goal, g_age = d.content[li, :N], d.age[li]
                parts = [
                    reports[b, r],
                    patches[b, r].ravel(),
                    masks[r].ravel(),
                    goal,
                    [min(g_age, AGE_CAP) / AGE_CAP],
                    np.eye(N_MODES)[int(s.mode)],
                    np.eye(N)[r],
                ]
                out[b, r] = np.concatenate(parts)
        return out

    def _crew_features(
        self, slots: list[EpisodeSlot], reports: np.ndarray, patches: np.ndarray, drop_masks: list[np.ndarray]
    ) -> np.ndarray:
        B, N, C = len(slots), self.N, self.C
        out = np.zeros((B, N, C, self.crew_feat_dim), np.float32)
        for b in range(B):
            masks = drop_masks[b].astype(np.float32)
            for r in range(N):
                for c in range(C):
                    out[b, r, c] = np.concatenate([patches[b, r, c].ravel(), masks[r, c].ravel(), reports[b, r]])
        return out

    def _regional_messages(
        self, msg: torch.Tensor, w: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gather peer messages per receiving region into [B, N, M, ...] padded tensors."""
        B = msg.shape[0]
        ci = self.content_idx
        msgs = torch.zeros(B, self.N, self.M, REPORT_DIM, device=self.device)
        ws = torch.zeros(B, self.N, self.M, device=self.device)
        vs = torch.zeros(B, self.N, self.M, dtype=torch.bool, device=self.device)
        for r in range(self.N):
            for k, q in enumerate(self.peers[r]):
                pos = int(np.searchsorted(ci, self.peer_idx[(q, r)]))
                msgs[:, r, k], ws[:, r, k], vs[:, r, k] = msg[:, pos], w[:, pos], valid[:, pos]
        return msgs, ws, vs

    def _sensitivity(
        self,
        regional_obs: np.ndarray,
        crew_feat: np.ndarray,
        extra_c: np.ndarray,
        decide: bool,
        msg: torch.Tensor,
        w_base: torch.Tensor,
        valid: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. (7) for every content link: leave-one-out KL times the tail cost of the induced decision.

        Both policies in the KL use the reference weights ``w_base`` (the kappa = 0 gate); the
        leave-one-out policy sets sender j's weight to zero. The tail cost is the expected
        CVaR cost of the action the receiver would take with the message, using only
        receiver-local inputs: per-crew Q^c on local features for regions, and the
        coordinator-local critic evaluated at the expected allocation for the coordinator.
        """
        cfg = self.cfg
        B, N, M, C = msg.shape[0], self.N, self.M, self.C
        alpha = cfg.train.cvar_alpha
        ci = self.content_idx
        psi = torch.zeros(B, len(ci), device=self.device)

        # Regional receivers: KL over crew heads; tail = sum_c E_{a ~ pi_c(.|M)} CVaR_alpha Q^c(h_c, a).
        msgs_r, w_r, valid_r = self._regional_messages(msg, w_base, valid)
        obs = self._t(regional_obs).view(B * N, -1)
        full = self.regional(obs, msgs_r.view(B * N, M, -1), w_r.view(B * N, M), valid_r.view(B * N, M))
        q_tail = self.crew_q.cvar(self.crew_q(self._t(crew_feat)), alpha)  # [B, N, C, A]
        probs = torch.softmax(full, -1).view(B, N, C, -1)
        tail_r = (probs * q_tail).sum(-1).sum(-1)  # [B, N]
        for k in range(M):
            w_k = w_r.clone()
            w_k[:, :, k] = 0.0
            loo = self.regional(obs, msgs_r.view(B * N, M, -1), w_k.view(B * N, M), valid_r.view(B * N, M))
            kl = categorical_kl(full, loo).sum(-1).view(B, N)
            for r in range(N):
                if k < len(self.peers[r]):
                    pos = int(np.searchsorted(ci, self.peer_idx[(self.peers[r][k], r)]))
                    psi[:, pos] = decision_sensitivity(kl[:, r], tail_r[:, r])

        # Coordinator receiver: KL over the allocation distribution; tail = CVaR of the coordinator-local
        # cost critic at the allocation the policy would make (expected drop shares), or at the
        # allocation in force between decision steps. No global state is used at execution.
        up_pos = np.searchsorted(ci, self.uplink_idx)
        extra = self._t(extra_c)
        slots_c, w_c, v_c = msg[:, up_pos], w_base[:, up_pos], valid[:, up_pos]
        full_c = self.coordinator(slots_c, w_c, v_c, extra)
        extra_alloc = extra.clone()
        if decide:
            extra_alloc[:, N_MODES : N_MODES + N] = torch.softmax(full_c, -1)
        tail_c = self.coord_cost.cvar(self.coord_cost(self.coord_local(slots_c, w_c, v_c, extra_alloc)), alpha)
        for j in range(N):
            w_j = w_c.clone()
            w_j[:, j] = 0.0
            kl = categorical_kl(full_c, self.coordinator(slots_c, w_j, v_c, extra))
            psi[:, up_pos[j]] = decision_sensitivity(kl, tail_c)
        return psi

    def _apply_mode(
        self,
        b: int,
        s: EpisodeSlot,
        w: np.ndarray,
        msg: np.ndarray,
        valid: torch.Tensor,
        t: int,
        up_pos: np.ndarray,
    ) -> int:
        """Modify weights and content in place for slot ``b`` according to its mode; return query count."""
        mc = self.cfg.method
        ci = self.content_idx
        v = valid[b].cpu().numpy()
        trusted_now = v & (w[b] >= 0.5)
        s.trusted[ci[trusted_now]] = msg[b, trusted_now]
        s.trusted_valid[ci[trusted_now]] = True
        if s.mode == Mode.NORMAL or mc.modes == "none":
            return 0
        if s.mode == Mode.FALLBACK:
            peer_pos = [p for p in range(len(ci)) if p not in set(up_pos.tolist())]
            w[b, peer_pos] = 0.0
            return 0
        # SELECTIVE and VERIFY: suspicious messages are replaced by the last trusted content.
        suspicious = v & (w[b] < 0.5)
        for p in np.flatnonzero(suspicious):
            if s.trusted_valid[ci[p]]:
                msg[b, p] = s.trusted[ci[p]]
                w[b, p] = 0.5
            else:
                w[b, p] = 0.0
        if s.mode != Mode.VERIFY or not mc.verify_mode:
            return 0
        # VERIFY: each neighbor of a suspicious uplink sender replies with its own border view of
        # the sender's region. A view can only bound the report from below (it covers part of the
        # region): a claim is contradicted if it reports fewer burning cells or a lower maximum
        # spread probability than the neighbor itself sees. A strict majority of consistent replies
        # restores the message at full weight; otherwise the selective treatment above stands.
        queries = 0
        area = s.env.region_h * s.env.region_w
        for j, p in enumerate(up_pos):
            if not suspicious[p]:
                continue
            received = s.channel.held[ci[p], :REPORT_DIM]
            replies = []
            for q in self.peers[j]:
                queries += 1
                rep = s.channel.verification_reply(t, q, s.env.border_view(q, j))
                if rep is not None:
                    replies.append(rep)
            if not replies:
                continue
            consistent = [
                received[0] * area >= r_[0] - VERIFY_CELL_TOL and received[2] >= r_[1] - VERIFY_SPREAD_TOL
                for r_ in replies
            ]
            if sum(consistent) > len(consistent) / 2:
                msg[b, p], w[b, p] = received, 1.0
        return queries
