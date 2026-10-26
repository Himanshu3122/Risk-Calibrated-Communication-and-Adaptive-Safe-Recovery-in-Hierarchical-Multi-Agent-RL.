"""Neural networks of RiCoR and its baselines.

Paper-specified elements: GRU channel estimator, evidential credibility head, multi-head
gated attention (Eq. 9), CVaR cost critic, mode selector. Layer widths follow Table III
where given; everything else is assumption A6 (``docs/assumptions.md``).
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from ricor.core.modes import N_MODES, SUMMARY_DIM
from ricor.envs.hierfire import N_CREW_ACTIONS, REPORT_DIM

META_DIM = 3  # arrived, age, gap


def mlp(in_dim: int, hidden: int, out_dim: int, layers: int = 2) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class ChannelEstimator(nn.Module):
    """GRU over link metadata eta_{1:t} producing logit(p_ch) (Sec. IV-A)."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(META_DIM, hidden)
        self.head = nn.Linear(hidden, 1)
        self.hidden_size = hidden

    def step(self, meta: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """meta [K, 3], h [K, H] -> (logit [K], h' [K, H])."""
        h = self.cell(meta, h)
        return self.head(h).squeeze(-1), h

    def sequence(self, meta: torch.Tensor) -> torch.Tensor:
        """meta [T, K, 3] -> logits [T, K], starting from a zero state."""
        h = meta.new_zeros(meta.shape[1], self.hidden_size)
        out = []
        for t in range(meta.shape[0]):
            logit, h = self.step(meta[t], h)
            out.append(logit)
        return torch.stack(out)


class CredibilityNet(nn.Module):
    """Evidential credibility head: non-negative evidence for and against (Eq. 4)."""

    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = mlp(in_dim, hidden, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ev = F.softplus(self.net(x))
        return ev[..., 0], ev[..., 1]


class GatedAttention(nn.Module):
    """Eq. (9): scaled dot-product attention whose weights are multiplied by gate w.

    Implemented as adding log(w) to the logits, with an always-valid learned null slot so
    that a receiver with every message gated out still has a defined context.
    """

    def __init__(self, query_dim: int, msg_dim: int, heads: int) -> None:
        super().__init__()
        if msg_dim % heads:
            raise ValueError("message_dim must be divisible by attention_heads")
        self.heads, self.dk = heads, msg_dim // heads
        self.q = nn.Linear(query_dim, msg_dim)
        self.k = nn.Linear(msg_dim, msg_dim)
        self.v = nn.Linear(msg_dim, msg_dim)
        self.null_k = nn.Parameter(torch.zeros(msg_dim))
        self.null_v = nn.Parameter(torch.zeros(msg_dim))
        self.out = nn.Linear(msg_dim, msg_dim)

    def forward(self, query: torch.Tensor, msgs: torch.Tensor, w: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """query [B, Q], msgs [B, M, D], w and valid [B, M] -> context [B, D]."""
        B, M, D = msgs.shape
        k = torch.cat([self.null_k.expand(B, 1, D), self.k(msgs)], 1)
        v = torch.cat([self.null_v.expand(B, 1, D), self.v(msgs)], 1)
        logw = torch.log(w.clamp_min(1e-6)) + torch.where(valid, 0.0, -1e9)
        logw = torch.cat([w.new_zeros(B, 1), logw], 1)
        q = self.q(query).view(B, self.heads, 1, self.dk)
        k = k.view(B, M + 1, self.heads, self.dk).transpose(1, 2)
        v = v.view(B, M + 1, self.heads, self.dk).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)).squeeze(2) / math.sqrt(self.dk) + logw.unsqueeze(1)
        attn = torch.softmax(logits, -1)
        ctx = (attn.unsqueeze(2) @ v).squeeze(2).reshape(B, D)
        return self.out(ctx)


class RegionalPolicy(nn.Module):
    """Shared regional policy pi^i(a | o, c, g, mu): one categorical head per crew."""

    def __init__(self, obs_dim: int, n_crews: int, hidden: int, msg_dim: int, heads: int) -> None:
        super().__init__()
        self.n_crews = n_crews
        self.encoder = mlp(obs_dim, hidden, hidden, layers=1)
        self.msg_enc = nn.Linear(REPORT_DIM, msg_dim)
        self.attn = GatedAttention(hidden, msg_dim, heads)
        self.head = mlp(hidden + msg_dim, hidden, n_crews * N_CREW_ACTIONS, layers=1)

    def forward(self, obs: torch.Tensor, msgs: torch.Tensor, w: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.encoder(obs))
        c = self.attn(h, self.msg_enc(msgs), w, valid)
        return self.head(torch.cat([h, c], -1)).view(-1, self.n_crews, N_CREW_ACTIONS)


class CoordinatorPolicy(nn.Module):
    """Coordinator pi^0: per-region scores from gated report slots; each drop is a categorical.

    Allocation needs per-region information, so the coordinator scores one gated slot per
    regional report (pointer-style attention) instead of pooling them (assumption A6).
    """

    def __init__(self, extra_dim: int, hidden: int, msg_dim: int) -> None:
        super().__init__()
        self.msg_enc = nn.Linear(REPORT_DIM, msg_dim)
        self.score = mlp(msg_dim + 2 + extra_dim, hidden, 1)

    def forward(self, slots: torch.Tensor, w: torch.Tensor, valid: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        """slots [B, N, 16], w/valid [B, N], extra [B, E] -> logits over regions [B, N]."""
        B, N, _ = slots.shape
        vf = valid.float()
        z = self.msg_enc(slots) * (w * vf).unsqueeze(-1)
        feats = torch.cat([z, w.unsqueeze(-1), vf.unsqueeze(-1), extra.unsqueeze(1).expand(B, N, -1)], -1)
        return self.score(feats).squeeze(-1)


class QuantileCritic(nn.Module):
    """Distributional critic: quantiles of a return or cost-to-go (assumption A11)."""

    def __init__(self, in_dim: int, hidden: int, n_quantiles: int, n_actions: int = 1) -> None:
        super().__init__()
        self.n_q, self.n_actions = n_quantiles, n_actions
        self.net = mlp(in_dim, hidden, n_actions * n_quantiles)
        taus = (torch.arange(n_quantiles, dtype=torch.float32) + 0.5) / n_quantiles
        self.register_buffer("taus", taus)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        return out.view(*x.shape[:-1], self.n_actions, self.n_q) if self.n_actions > 1 else out

    def cvar(self, quantiles: torch.Tensor, alpha: float) -> torch.Tensor:
        """Upper-tail CVaR_alpha of a cost distribution: mean of quantiles above level alpha."""
        upper = (self.taus >= alpha).nonzero().flatten()
        idx = upper if len(upper) else torch.tensor([self.n_q - 1], device=quantiles.device)
        return quantiles.sort(-1).values[..., idx].mean(-1)


class ValueCritic(nn.Module):
    def __init__(self, in_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = mlp(in_dim, hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ModeSelector(nn.Module):
    """pi^mu over {N, S, V, F} from the summary xi_t and the current mode, with a value head."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.body = mlp(SUMMARY_DIM + N_MODES, hidden, hidden, layers=1)
        self.pi = nn.Linear(hidden, N_MODES)
        self.v = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.body(x))
        logits = self.pi(h).masked_fill(~mask, -1e9)
        return logits, self.v(h).squeeze(-1)


class MissingInfoPredictor(nn.Module):
    """HPSF baseline: recurrent prediction of a sender's current report from stale deliveries."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.cell = nn.GRUCell(REPORT_DIM + 1, hidden)
        self.head = nn.Linear(hidden, REPORT_DIM)
        self.hidden_size = hidden

    def step(self, held: torch.Tensor, age: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.cell(torch.cat([held, age.unsqueeze(-1)], -1), h)
        return held + self.head(h), h
