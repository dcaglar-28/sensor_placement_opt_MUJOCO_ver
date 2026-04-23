"""
sensor_opt/loss/loss.py

Default loss (mode=`default`):
  L = α·collision_rate + β·blind_spot_fraction + γ·cost_penalty

Obstacle / latency research mode (mode=`obstacle_latency`):
  L = α·(p95(t_det) / t_det_max_s) + β·collision_rate
  (optionally + γ·cost_penalty if you set γ>0)

In `default` mode, the main terms are naturally in [0, 1] (plus hardware penalty).
In `obstacle_latency` mode, the normalized p95 term is clipped to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - fallback for environments without JAX
    jnp = None

from sensor_opt.encoding.config import SensorConfig


@dataclass
class EvalMetrics:
    """Raw metrics returned by the inner-loop evaluator."""
    collision_rate: float
    blind_spot_fraction: float
    mean_goal_success: float
    n_episodes: int
    # Optional obstacle / latency fields (Isaac Sim / Isaac Lab use-case)
    # When unset, they default to 0.0 and do not affect legacy loss modes.
    t_det_s: float = 0.0
    t_det_s_p95: float = 0.0
    episode_time_s: float = 0.0
    safety_success: float = 0.0


@dataclass
class LossResult:
    """Fully decomposed loss for logging and analysis."""
    total: float
    collision_term: float
    blind_term: float
    cost_term: float
    cost_usd: float
    n_active_sensors: int
    config_summary: str
    hardware_penalty_term: float = 0.0
    objectives: Dict[str, float] | None = None


def loss_weight_dict(loss_cfg: dict) -> dict:
    """
    Normalize YAML `loss:` config into the `weights` dict expected by `compute_loss`.

    This keeps the call sites in outer loops and search methods consistent.
    """
    w = {
        "alpha": float(loss_cfg.get("alpha", 0.0)),
        "beta": float(loss_cfg.get("beta", 0.0)),
        "gamma": float(loss_cfg.get("gamma", 0.0)),
    }
    if "t_det_max_s" in loss_cfg and loss_cfg.get("t_det_max_s") is not None:
        w["t_det_max_s"] = float(loss_cfg["t_det_max_s"])
    return w


def compute_loss(
    metrics: EvalMetrics,
    config: SensorConfig,
    sensor_models: dict,
    weights: dict,
    max_cost_usd: float = 10_000.0,
    hardware_constraints: dict | None = None,
    loss_mode: str = "default",
) -> LossResult:
    xp = _array_lib()

    alpha = weights["alpha"]
    beta  = weights["beta"]
    gamma = weights["gamma"]
    t_max = float((weights or {}).get("t_det_max_s", 30.0))

    if loss_mode == "obstacle_latency":
        # User-specified research objective:
        #   Loss = alpha * p95(t_det) + beta * CollisionRate
        # p95 is normalized to [0,1] by dividing by t_det_max_s (clipped).
        p95 = float(xp.clip(float(getattr(metrics, "t_det_s_p95", 0.0)) / max(t_max, 1e-6), 0.0, 1.0))
        col = float(xp.clip(float(metrics.collision_rate), 0.0, 1.0))

        collision_term = float(beta * col)
        blind_term = float(alpha * p95)  # beta slot historically carried "perception" term; here it's latency
        # Intentionally disable cost in this mode unless user sets gamma>0
        cost_usd = 0.0
        cost_penalty = 0.0
        cost_term = float(gamma * cost_penalty)
    else:
        collision_term = float(alpha * xp.clip(float(metrics.collision_rate), 0.0, 1.0))
        blind_term = float(beta * xp.clip(float(metrics.blind_spot_fraction), 0.0, 1.0))

        cost_usd = _compute_effective_cost(config, sensor_models)
        cost_penalty = float(xp.clip(cost_usd / max_cost_usd, 0.0, 1.0))
        cost_term = float(gamma * cost_penalty)

    hardware_penalty_term = _compute_hardware_penalty(config, sensor_models, hardware_constraints)
    total = collision_term + blind_term + cost_term + hardware_penalty_term

    if not config.active_sensors():
        if loss_mode == "obstacle_latency":
            # No sensors => cannot perceive => worst-case for both latency + collision terms (bounded, interpretable)
            total = float(alpha) + float(beta)
        else:
            total = 1.0

    if loss_mode == "obstacle_latency":
        objectives = {
            # For legacy scalarizers that sum objectives["collision"] + ["blind_spot"] + ...:
            "collision": float(_clamp(metrics.collision_rate)),
            "blind_spot": float(_clamp(float(getattr(metrics, "t_det_s_p95", 0.0)) / max(t_max, 1e-6))),
            "cost": 0.0,
            "hardware": 0.0,
            "t_det_s_p95": float(getattr(metrics, "t_det_s_p95", 0.0)),
            "safety_success": float(_clamp(getattr(metrics, "safety_success", 0.0))),
        }
    else:
        objectives = {
            "collision": float(_clamp(metrics.collision_rate)),
            "blind_spot": float(_clamp(metrics.blind_spot_fraction)),
            "cost": cost_penalty,
            "hardware": float(_clamp(hardware_penalty_term)),
        }

    total_out = float(total) if loss_mode == "obstacle_latency" else float(_clamp(total))

    return LossResult(
        total=total_out,
        collision_term=collision_term,
        blind_term=blind_term,
        cost_term=cost_term,
        hardware_penalty_term=hardware_penalty_term,
        cost_usd=cost_usd,
        n_active_sensors=len(config.active_sensors()),
        config_summary=config.summary(),
        objectives=objectives,
    )


def _array_lib():
    """Use JAX arrays when available, otherwise NumPy."""
    return jnp if jnp is not None else np


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    xp = _array_lib()
    return float(xp.clip(v, lo, hi))


def _compute_effective_cost(config: SensorConfig, sensor_models: dict) -> float:
    active = config.active_sensors()
    if not active:
        return 0.0

    type_instance: Dict[str, int] = {}
    total = 0.0

    for sensor in active:
        model = sensor_models.get(sensor.sensor_type, {})
        base_cost = float(model.get("cost_usd", 0.0))
        idx = type_instance.get(sensor.sensor_type, 0)
        discount = 1.0 - 0.05 * min(idx, 3)
        total += base_cost * discount
        total += 50.0
        type_instance[sensor.sensor_type] = idx + 1

    return total


def _compute_hardware_penalty(
    config: SensorConfig,
    sensor_models: dict,
    hardware_constraints: dict | None,
) -> float:
    if not hardware_constraints:
        return 0.0

    active = config.active_sensors()
    if not active:
        return 0.0

    total_compute = 0.0
    total_memory = 0.0
    total_latency = 0.0
    for sensor in active:
        model = sensor_models.get(sensor.sensor_type, {})
        total_compute += float(model.get("compute_tops", 2.0))
        total_memory += float(model.get("memory_gb", 0.3))
        total_latency += float(model.get("latency_ms", 10.0))

    penalty = 0.0
    compute_limit = float(hardware_constraints.get("compute_limit_tops", 1e9))
    memory_limit = float(hardware_constraints.get("memory_limit_gb", 1e9))
    latency_budget = float(hardware_constraints.get("latency_budget_ms", 1e9))

    if total_compute > compute_limit:
        penalty += (total_compute - compute_limit) / max(compute_limit, 1e-6)
    if total_memory > memory_limit:
        penalty += (total_memory - memory_limit) / max(memory_limit, 1e-6)
    if total_latency > latency_budget:
        penalty += 2.0 * (total_latency - latency_budget) / max(latency_budget, 1e-6)

    return float(_clamp(penalty))