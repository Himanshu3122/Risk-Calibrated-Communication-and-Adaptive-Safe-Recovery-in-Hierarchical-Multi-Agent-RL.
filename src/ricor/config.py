"""Typed experiment configuration loaded from YAML.

An experiment file may list other YAML files under ``include`` (relative to itself).
Included files are merged first, in order, and the experiment's own keys override them.
Unknown keys raise :class:`ConfigError`, so typos never silently change an experiment.
"""

from __future__ import annotations

import copy
import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised for malformed or inconsistent configuration."""


@dataclass
class EnvConfig:
    """HierFire environment (paper Sec. VI-A)."""

    height: int = 48
    width: int = 48
    region_rows: int = 2
    region_cols: int = 4
    crews_per_region: int = 3
    n_drops: int = 6
    coordinator_period: int = 5  # K
    episode_length: int = 400
    assets_min: int = 12
    assets_max: int = 20
    ignitions_min: int = 2
    ignitions_max: int = 4
    entrapment_threshold: float = 0.6
    drop_radius: int = 1
    drop_latency: int = 2  # steps between allocation and drop landing (assumption A3)
    patch_radius: int = 2
    base_spread: float = 0.12
    burn_duration: int = 6
    suppression_amount: float = 0.5
    wind_speed_max: float = 1.0
    halo_width: int = 3  # cells of a neighboring region a regional agent can observe (verification)

    @property
    def n_regions(self) -> int:
        return self.region_rows * self.region_cols

    def validate(self) -> None:
        if self.height % self.region_rows or self.width % self.region_cols:
            raise ConfigError("grid size must be divisible by the region layout")
        if self.n_regions < 2:
            raise ConfigError("at least two regions are required")
        if not 0.0 < self.entrapment_threshold < 1.0:
            raise ConfigError("entrapment_threshold must lie in (0, 1)")


@dataclass
class ScenarioConfig:
    """Procedural map library and seed split (paper Sec. VI-B)."""

    n_maps: int = 6000
    n_train: int = 4200
    n_val: int = 600
    n_test: int = 1200
    base_seed: int = 12345

    def validate(self) -> None:
        if self.n_train + self.n_val + self.n_test != self.n_maps:
            raise ConfigError("train/val/test sizes must sum to n_maps")


@dataclass
class ModelConfig:
    """Network sizes (paper Table III where given)."""

    message_dim: int = 32  # d_m
    gru_hidden: int = 128
    hidden: int = 128
    attention_heads: int = 4
    n_quantiles: int = 16
    credibility_hidden: int = 64


@dataclass
class MethodConfig:
    """Switches that define RiCoR, its baselines, and its ablations."""

    name: str = "ricor"
    gating: bool = True  # use reliability weights at all
    split_reliability: bool = True  # channel x content factorization (Eq. 6)
    uncertainty: bool = True  # evidential LCB (Eqs. 4-6); False -> point estimate
    lambda_u: float = 1.0
    kappa: float = 1.0  # risk sensitivity (Eq. 8); 0 -> risk-agnostic
    tau0: float = 0.5
    gate_temperature: float = 0.1  # T_g
    modes: str = "learned"  # learned | fixed | none
    verify_mode: bool = True
    shield: bool = True
    message_dropout: float = 0.0  # H-MAPPO+Drop
    message_noise: float = 0.0
    predict_missing: bool = False  # HPSF recurrent predictor
    train_with_faults: bool = True
    # Reference bounds and the trust-model baseline added in revision.
    oracle_gate: bool = False  # upper bound: gate weight = true reliability label (privileged by design)
    no_comm: bool = False  # lower bound: every received report and goal is discarded
    reputation: bool = False  # Beta-reputation weighting (BRS/TRAVOS-style) instead of learned reliability
    reputation_decay: float = 0.95  # forgetting factor of the Beta evidence counts
    reputation_tol: float = 0.2  # max step-to-step change of threat fields counted as consistent
    reputation_wind_tol: float = 0.05  # max deviation of the (global) wind fields counted as consistent
    hysteresis: int = 5  # H
    switch_cost: float = 0.05  # c_sw
    query_cost: float = 0.02  # c_v
    # Fixed-threshold switching rule (Ours-fixed ablation)
    fixed_tau: float = 0.5
    fixed_psi_v: float = 0.5
    fixed_sigma_v: float = 0.15
    fixed_outage: int = 8

    def validate(self) -> None:
        if self.modes not in {"learned", "fixed", "none"}:
            raise ConfigError(f"unknown modes setting {self.modes!r}")
        if not 0.0 <= self.message_dropout < 1.0:
            raise ConfigError("message_dropout must lie in [0, 1)")


@dataclass
class TrainConfig:
    """PPO-Lagrangian training (paper Table III)."""

    total_steps: int = 10_000_000
    n_envs: int = 32
    rollout_length: int = 400
    actor_lr: float = 5e-4
    critic_lr: float = 1e-3
    lagrange_lr: float = 5e-3  # eta_lambda
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ppo_epochs: int = 10
    minibatches: int = 4
    entropy_coef: float = 0.01  # c_e
    value_coef: float = 0.5  # c_val (not reported in paper; assumption A7)
    cost_coef: float = 0.5  # c_c (not reported; assumption A7)
    reliability_coef: float = 1.0  # c_r (not reported; assumption A7)
    beta_kl_max: float = 0.05
    cvar_alpha: float = 0.9
    cost_budget: float = 0.5  # d
    lambda_max: float = 20.0
    max_grad_norm: float = 0.5
    curriculum: list[float] = field(default_factory=lambda: [0.2, 0.4, 0.4])
    checkpoint_every: int = 0  # iterations; 0 -> only final
    device: str = "cpu"

    def validate(self) -> None:
        if len(self.curriculum) != 3 or abs(sum(self.curriculum) - 1.0) > 1e-6:
            raise ConfigError("curriculum must be three fractions summing to 1")
        if not 0.0 < self.cvar_alpha < 1.0:
            raise ConfigError("cvar_alpha must lie in (0, 1)")


@dataclass
class EvalConfig:
    """Evaluation protocol (paper Sec. VI-E)."""

    conditions: list[str] = field(
        default_factory=lambda: ["clean", "F1", "F2", "F3", "F4", "F5", "F6", "C1", "C2", "C3", "U1", "U2", "U3"]
    )
    episodes_per_condition: int = 1200
    loss_sweep: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    recovery_condition: str = "C2"
    recovery_window: list[int] = field(default_factory=lambda: [100, 250])
    recovery_episodes: int = 50
    recovery_smoothing: int = 10
    ece_bins: int = 15
    bootstrap_samples: int = 10000
    split: str = "test"

    def validate(self) -> None:
        if len(self.recovery_window) != 2 or self.recovery_window[0] >= self.recovery_window[1]:
            raise ConfigError("recovery_window must be [start, end] with start < end")


@dataclass
class ExperimentConfig:
    """Top-level configuration."""

    name: str = "experiment"
    seed: int = 0
    output_dir: str = "runs"
    env: EnvConfig = field(default_factory=EnvConfig)
    scenarios: ScenarioConfig = field(default_factory=ScenarioConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    method: MethodConfig = field(default_factory=MethodConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def validate(self) -> None:
        for part in (self.env, self.scenarios, self.method, self.train, self.eval):
            part.validate()

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


_SECTIONS = {
    "env": EnvConfig,
    "scenarios": ScenarioConfig,
    "model": ModelConfig,
    "method": MethodConfig,
    "train": TrainConfig,
    "eval": EvalConfig,
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ConfigError(f"circular include: {path}")
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a mapping")
    merged: dict[str, Any] = {}
    for inc in data.pop("include", []) or []:
        merged = deep_merge(merged, _read_yaml(path.parent / inc, seen + (path,)))
    return deep_merge(merged, data)


def _build(cls: type, values: dict[str, Any], where: str) -> Any:
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ConfigError(f"unknown keys in {where}: {sorted(unknown)}")
    return cls(**values)


def from_dict(data: dict[str, Any]) -> ExperimentConfig:
    """Build and validate an :class:`ExperimentConfig` from a plain mapping."""
    data = dict(data)
    kwargs: dict[str, Any] = {}
    for section, cls in _SECTIONS.items():
        kwargs[section] = _build(cls, data.pop(section, {}) or {}, section)
    cfg = _build(ExperimentConfig, {**data, **kwargs}, "experiment")
    cfg.validate()
    return cfg


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> ExperimentConfig:
    """Load an experiment YAML (with includes) and apply optional overrides."""
    data = _read_yaml(Path(path))
    if overrides:
        data = deep_merge(data, overrides)
    return from_dict(data)


def parse_override(text: str) -> dict[str, Any]:
    """Parse ``section.key=value`` into a nested dict; the value is parsed as YAML."""
    if "=" not in text:
        raise ConfigError(f"override must look like a.b=value, got {text!r}")
    dotted, raw = text.split("=", 1)
    keys = [k for k in dotted.strip().split(".") if k]
    if not keys:
        raise ConfigError(f"empty override key in {text!r}")
    value: Any = yaml.safe_load(raw)
    for key in reversed(keys):
        value = {key: value}
    return value
