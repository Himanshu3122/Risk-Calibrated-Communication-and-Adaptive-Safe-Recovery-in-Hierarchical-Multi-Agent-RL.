"""Centralized PPO-Lagrangian training with the fault curriculum (paper Sec. V-E, V-F, Eqs. 12-14)."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ricor.comm.faults import sample_training_spec
from ricor.config import ExperimentConfig, from_dict
from ricor.core.reliability import reliability_loss
from ricor.scenarios import split_seeds
from ricor.system import RiCoRSystem
from ricor.utils import environment_metadata, get_logger, set_seed, write_json

log = get_logger("ricor.training")


def gae(rewards: torch.Tensor, values: torch.Tensor, gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation over [T, B]; episodes end at T without bootstrapping."""
    T = rewards.shape[0]
    adv = torch.zeros_like(rewards)
    last = torch.zeros_like(rewards[0])
    for t in reversed(range(T)):
        next_v = values[t + 1] if t + 1 < T else torch.zeros_like(values[t])
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    return adv, adv + values


def quantile_huber(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
    """Quantile regression Huber loss; ``pred`` [..., Q], ``target`` [...]."""
    u = target.unsqueeze(-1) - pred
    huber = torch.where(u.abs() <= kappa, 0.5 * u.pow(2), kappa * (u.abs() - 0.5 * kappa))
    return ((taus - (u < 0).float()).abs() * huber).mean()


def empirical_cvar(x: torch.Tensor, alpha: float) -> torch.Tensor:
    """Mean of the values at or above the alpha-quantile (upper tail)."""
    q = torch.quantile(x, alpha)
    return x[x >= q].mean()


class Trainer:
    """Runs the curriculum, PPO updates, Lagrange updates, calibration, and checkpointing."""

    def __init__(self, cfg: ExperimentConfig, run_dir: str | Path) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        set_seed(cfg.seed)
        self.system = RiCoRSystem(cfg).to(cfg.train.device)
        tc = cfg.train
        s = self.system
        actor = [*s.regional.parameters(), *s.coordinator.parameters(), *s.mode_selector.parameters()]
        critic = [
            *s.value.parameters(),
            *s.cost_critic.parameters(),
            *s.coord_cost.parameters(),
            *s.crew_q.parameters(),
            *s.channel_est.parameters(),
            *s.credibility.parameters(),
            *s.predictor.parameters(),
        ]
        self.opt = torch.optim.Adam([{"params": actor, "lr": tc.actor_lr}, {"params": critic, "lr": tc.critic_lr}])
        self.splits = split_seeds(cfg.scenarios)
        self.rng = np.random.default_rng(cfg.seed)
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ loop
    def stage(self, progress: float) -> int:
        if not self.cfg.method.train_with_faults:
            return 0
        c = self.cfg.train.curriculum
        return 0 if progress < c[0] else (1 if progress < c[0] + c[1] else 2)

    def train(self) -> Path:
        cfg, tc = self.cfg, self.cfg.train
        steps_per_iter = tc.n_envs * cfg.env.episode_length
        iterations = max(1, math.ceil(tc.total_steps / steps_per_iter))
        log.info(
            "training %s seed=%d for %d iterations (%d env steps each)",
            cfg.method.name,
            cfg.seed,
            iterations,
            steps_per_iter,
        )
        for it in range(iterations):
            progress = it / iterations
            stage = self.stage(progress)
            maps = self.rng.choice(self.splits["train"], size=tc.n_envs).tolist()
            faults = [sample_training_spec(stage, self.rng, self.system.N) for _ in range(tc.n_envs)]
            ch_seeds = self.rng.integers(0, 2**31 - 1, size=tc.n_envs).tolist()
            t0 = time.perf_counter()
            self.system.eval()
            roll = self.system.run_episodes(maps, faults, ch_seeds, stochastic=True, training=True)
            self.system.train()
            stats = self.update(roll.data, progress)
            stats.update(
                iteration=it,
                stage=stage,
                seconds=time.perf_counter() - t0,
                apr=float(np.mean([m["apr"] for m in roll.episode_metrics])),
                sv=float(np.mean([m["sv"] for m in roll.episode_metrics])),
                ret=float(np.mean([m["return"] for m in roll.episode_metrics])),
            )
            self.history.append(stats)
            log.info(
                "it %d/%d stage %d | return %.3f apr %.1f sv %.2f lambda %.3f | %.1fs",
                it + 1,
                iterations,
                stage,
                stats["ret"],
                stats["apr"],
                stats["sv"],
                stats["lambda"],
                stats["seconds"],
            )
            if tc.checkpoint_every and (it + 1) % tc.checkpoint_every == 0:
                self.save(self.run_dir / f"checkpoint_it{it + 1}.pt")
        self.calibrate()
        path = self.save(self.run_dir / "checkpoint.pt")
        write_json(self.run_dir / "train_history.json", self.history)
        return path

    # ------------------------------------------------------------------ update
    def update(self, d: dict[str, torch.Tensor], progress: float) -> dict[str, float]:
        cfg, tc, s = self.cfg, self.cfg.train, self.system
        dev = s.device
        d = {k: v.to(dev) for k, v in d.items()}
        T, B = d["reward"].shape
        lam = float(s.lagrange)
        with torch.no_grad():
            values = s.value(d["gstate"])
            cost_q = s.cost_critic(d["gstate"])
            adv_r, ret_r = gae(d["reward"], values, tc.gamma, tc.gae_lambda)
            adv_c, ret_c = gae(d["cost"], cost_q.mean(-1), tc.gamma, tc.gae_lambda)
            adv = adv_r - lam * adv_c
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
            adv_m, ret_m = gae(d["mode_reward"], d["mode_v"].float(), tc.gamma, tc.gae_lambda)
            adv_m = (adv_m - adv_m.mean()) / (adv_m.std() + 1e-8)
            # Discounted cost-to-go: the same quantity the cost critic stands for in GAE above.
            ret_cost_mc = torch.zeros_like(d["cost"])
            running = torch.zeros_like(d["cost"][0])
            for t in reversed(range(T)):
                running = d["cost"][t] + tc.gamma * running
                ret_cost_mc[t] = running

        flat = lambda x: x.reshape(T * B, *x.shape[2:])  # noqa: E731
        F = {
            k: flat(v)
            for k, v in d.items()
            if k not in ("meta", "cred_x", "y_ch", "y_cr", "has_c", "pred_held", "pred_age", "pred_target")
        }
        A, ARet, AM, RM, RC = flat(adv), flat(ret_r), flat(adv_m), flat(ret_m), flat(ret_cost_mc)
        next_idx = torch.arange(T * B, device=dev) + B
        terminal = next_idx >= T * B
        next_idx = next_idx.clamp(max=T * B - 1)
        n = T * B
        mb_size = max(1, n // tc.minibatches)
        stats: dict[str, list[float]] = {
            k: [] for k in ("pi_r", "pi_c", "pi_m", "v", "cost", "crew_q", "rel", "pred", "ent")
        }
        beta_kl = tc.beta_kl_max * min(1.0, progress * 2.0)
        for _ in range(tc.ppo_epochs):
            perm = torch.randperm(n, device=dev)
            for start in range(0, n, mb_size):
                idx = perm[start : start + mb_size]
                loss = self._ppo_losses(F, idx, A, ARet, AM, RM, RC, next_idx, terminal, stats)
                self._step(loss)
            # Reliability heads (and the HPSF predictor) unroll over the full episode: once per epoch.
            # Methods that never read the heads do not train them.
            mc = cfg.method
            loss = None
            if mc.gating and not (mc.reputation or mc.oracle_gate or mc.no_comm):
                rel_loss = self._reliability_loss(d, beta_kl)
                stats["rel"].append(float(rel_loss))
                loss = tc.reliability_coef * rel_loss
            if mc.predict_missing:
                pred_loss = self._predictor_loss(d)
                stats["pred"].append(float(pred_loss))
                loss = pred_loss if loss is None else loss + pred_loss
            if loss is not None:
                self._step(loss)

        # Eq. (14): projected gradient ascent on the Lagrange multiplier.
        ep_cost = d["cost"].sum(0)
        cvar = float(empirical_cvar(ep_cost, tc.cvar_alpha))
        new_lam = min(tc.lambda_max, max(0.0, lam + tc.lagrange_lr * (cvar - tc.cost_budget)))
        s.lagrange.fill_(new_lam)
        out = {k: float(np.mean(v)) if v else 0.0 for k, v in stats.items()}
        out.update(cvar_cost=cvar, mean_cost=float(ep_cost.mean()), **{"lambda": new_lam})
        return out

    def _step(self, loss: torch.Tensor) -> None:
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.system.parameters(), self.cfg.train.max_grad_norm)
        self.opt.step()

    def _ppo_losses(
        self,
        F: dict[str, torch.Tensor],
        idx: torch.Tensor,
        A: torch.Tensor,
        ARet: torch.Tensor,
        AM: torch.Tensor,
        RM: torch.Tensor,
        RC: torch.Tensor,
        next_idx: torch.Tensor,
        terminal: torch.Tensor,
        stats: dict[str, list[float]],
    ) -> torch.Tensor:
        tc, s = self.cfg.train, self.system
        clip = tc.clip
        N, M, C = s.N, s.M, s.C
        b = len(idx)

        def ppo(new_logp: torch.Tensor, old_logp: torch.Tensor, adv: torch.Tensor) -> torch.Tensor:
            ratio = torch.exp(new_logp - old_logp)
            return -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()

        # Regional policies (shared parameters, one ratio per agent).
        logits = s.regional(
            F["obs_r"][idx].reshape(b * N, -1),
            F["msgs_r"][idx].reshape(b * N, M, -1),
            F["w_r"][idx].reshape(b * N, M),
            F["valid_r"][idx].reshape(b * N, M),
        )
        dist = torch.distributions.Categorical(logits=logits)
        new_lp = dist.log_prob(F["act_r"][idx].reshape(b * N, C)).sum(-1)
        loss_r = ppo(new_lp, F["logp_r"][idx].reshape(-1), A[idx].repeat_interleave(N))
        entropy = dist.entropy().mean()
        stats["pi_r"].append(float(loss_r))
        stats["ent"].append(float(entropy))
        loss = loss_r - tc.entropy_coef * entropy

        # Coordinator at decision steps.
        dec = F["decided_c"][idx].bool()
        if dec.any():
            j = idx[dec]
            lg = s.coordinator(F["slots_c"][j], F["w_c"][j], F["valid_c"][j].bool(), F["extra_c"][j])
            n_drops = F["drops_c"].shape[-1]
            dist_c = torch.distributions.Categorical(logits=lg.unsqueeze(1).expand(len(j), n_drops, N))
            loss_c = ppo(dist_c.log_prob(F["drops_c"][j]).sum(-1), F["logp_c"][j], A[j])
            loss = loss + loss_c - tc.entropy_coef * dist_c.entropy().mean()
            stats["pi_c"].append(float(loss_c))

        # Mode selector (Eq. 10 reward) with its own value head.
        md = F["mode_decided"][idx].bool()
        if md.any():
            j = idx[md]
            lg, v = s.mode_selector(F["mode_x"][j], F["mode_mask"][j].bool())
            dist_m = torch.distributions.Categorical(logits=lg)
            loss_m = ppo(dist_m.log_prob(F["mode_a"][j]), F["mode_logp"][j], AM[j])
            loss = loss + loss_m + tc.value_coef * (v - RM[j]).pow(2).mean() - tc.entropy_coef * dist_m.entropy().mean()
            stats["pi_m"].append(float(loss_m))

        # Critics.
        v_loss = (s.value(F["gstate"][idx]) - ARet[idx]).pow(2).mean()
        c_loss = quantile_huber(s.cost_critic(F["gstate"][idx]), RC[idx], s.cost_critic.taus)
        c_loss = c_loss + quantile_huber(s.coord_cost(F["coord_x"][idx]), RC[idx], s.coord_cost.taus)
        q = s.crew_q(F["crew_feat"][idx])  # [b, N, C, A, Q]
        a_exec = F["crew_exec"][idx].long()
        q_sa = q.gather(-2, a_exec[..., None, None].expand(*a_exec.shape, 1, q.shape[-1])).squeeze(-2)
        with torch.no_grad():
            nq = s.crew_q(F["crew_feat"][next_idx[idx]])
            na = F["crew_exec"][next_idx[idx]].long()
            nq_sa = nq.gather(-2, na[..., None, None].expand(*na.shape, 1, nq.shape[-1])).squeeze(-2)
            cont = (~terminal[idx]).float().reshape(-1, 1, 1, 1)
            target = F["crew_cost"][idx].unsqueeze(-1) + tc.gamma * cont * nq_sa
        u = target.unsqueeze(-1) - q_sa.unsqueeze(-2)  # pairwise quantile TD
        taus = s.crew_q.taus.reshape(1, 1, 1, 1, -1)
        huber = torch.where(u.abs() <= 1.0, 0.5 * u.pow(2), u.abs() - 0.5)
        q_loss = ((taus - (u < 0).float()).abs() * huber).mean()
        stats["v"].append(float(v_loss))
        stats["cost"].append(float(c_loss))
        stats["crew_q"].append(float(q_loss))
        return loss + tc.value_coef * v_loss + tc.cost_coef * (c_loss + q_loss)

    def _reliability_loss(self, d: dict[str, torch.Tensor], beta_kl: float) -> torch.Tensor:
        """Eq. (12) over the whole rollout (the GRU is unrolled from the episode start)."""
        s = self.system
        T, B, L, _ = d["meta"].shape
        p_logit = s.channel_est.sequence(d["meta"].reshape(T, B * L, -1)).reshape(T, B, L)
        p_ch = torch.sigmoid(p_logit)
        e_pos, e_neg = s.credibility(d["cred_x"])
        alpha, beta = 1.0 + e_pos, 1.0 + e_neg
        has = d["has_c"].bool()
        y_cr = d["y_cr"].float()
        if not self.cfg.method.split_reliability:
            ci = torch.as_tensor(s.content_idx, device=y_cr.device)
            y_cr = y_cr * d["y_ch"].index_select(2, ci).float()
        ch_part = (d["y_ch"].float() - p_ch).pow(2).mean()
        if not has.any():
            return ch_part
        return ch_part + reliability_loss(
            p_ch.new_ones(int(has.sum())), p_ch.new_ones(int(has.sum())), alpha[has], beta[has], y_cr[has], beta_kl
        )

    def _predictor_loss(self, d: dict[str, torch.Tensor]) -> torch.Tensor:
        """HPSF: MSE between predicted and true sender reports on stale or missing links."""
        s = self.system
        T, B, Lc, D = d["pred_held"].shape
        h = d["pred_held"].new_zeros(B * Lc, s.predictor.hidden_size)
        losses = []
        for t in range(T):
            pred, h = s.predictor.step(d["pred_held"][t].reshape(B * Lc, D), d["pred_age"][t].reshape(-1).float(), h)
            losses.append((pred - d["pred_target"][t].reshape(B * Lc, D)).pow(2).mean())
        return torch.stack(losses).mean()

    # ------------------------------------------------------------------ calibration
    @torch.no_grad()
    def calibrate(self, episodes: int | None = None) -> dict[str, float]:
        """Temperature-scale p_ch and r_bar on validation maps (Guo et al. 2017) by grid search on NLL."""
        s = self.system
        mc = self.cfg.method
        if not mc.gating or mc.reputation or mc.oracle_gate or mc.no_comm:
            return {}  # no learned reliability heads in use
        n = episodes or min(self.cfg.train.n_envs, len(self.splits["val"]))
        maps = self.splits["val"][:n]
        faults = [sample_training_spec(2 if self.cfg.method.train_with_faults else 0, self.rng, s.N) for _ in maps]
        s.temp_ch.fill_(1.0)
        s.temp_cr.fill_(1.0)
        s.eval()
        roll = s.run_episodes(maps, faults, list(range(n)), stochastic=False, record_reliability=True)
        rec = roll.reliability_records
        grid = np.exp(np.linspace(np.log(0.25), np.log(4.0), 41))
        out = {}
        for key, label, buf in (("p_ch", "y_ch", s.temp_ch), ("r_bar", "y_cr", s.temp_cr)):
            p, y = rec[key], rec[label]
            if key == "r_bar" and not self.cfg.method.split_reliability:
                y = rec["y_ch"] * rec["y_cr"]  # the joint head is trained on the joint label
            if len(p) == 0 or y.min() == y.max():
                out[key] = 1.0
                continue
            logit = np.log(np.clip(p, 1e-5, 1 - 1e-5) / np.clip(1 - p, 1e-5, 1))
            nll = [
                float(-np.mean(y * -np.logaddexp(0, -logit / T) + (1 - y) * -np.logaddexp(0, logit / T))) for T in grid
            ]
            best = float(grid[int(np.argmin(nll))])
            buf.fill_(best)
            out[key] = best
        log.info("calibration temperatures: %s", out)
        return out

    # ------------------------------------------------------------------ checkpoints
    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"state_dict": self.system.state_dict(), "config": self.cfg.to_dict(), "metadata": environment_metadata()},
            path,
        )
        write_json(path.with_suffix(".json"), {"config": self.cfg.to_dict(), "metadata": environment_metadata()})
        log.info("saved checkpoint %s", path)
        return path


def load_system(path: str | Path, device: str = "cpu") -> tuple[RiCoRSystem, ExperimentConfig]:
    """Rebuild a system from a checkpoint written by :meth:`Trainer.save`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    cfg = from_dict(payload["config"])
    system = RiCoRSystem(cfg).to(device)
    system.load_state_dict(payload["state_dict"])
    system.eval()
    return system, cfg
