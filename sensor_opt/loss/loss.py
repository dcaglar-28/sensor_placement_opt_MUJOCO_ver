"""
sensor_opt/loss/loss.py

Default loss (mode=`default`):
  L = α·collision_rate + β·blind_spot_fraction + γ·cost_penalty

Obstacle / latency research mode (mode=`obstacle_latency`):
  L = α·(p95(t_det) / t_det_max_s) + β·collision_rate
  (optionally + γ·cost_penalty if you set γ>0)

MuJoCo hazard-detection mode (mode=`mujoco_tri`):
  L = α·(p95(t_det) / t_det_max_s) + β·detection_miss_rate
      + γ·cost_penalty  (+ hardware budget penalty, same as `default`)

In `default` mode, the main terms are naturally in [0, 1] (plus hardware penalty).
In `obstacle_latency` and `mujoco_tri` modes, the normalized p95 term is clipped to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import jax.numpy as jnp

from sensor_opt.encoding.config import SensorConfig


@dataclass
class EvalMetrics:
    """Raw metrics returned by the inner-loop evaluator."""
    collision_rate: float
    blind_spot_fraction: float
    mean_goal_success: float
    n_episodes: int
    # Optional obstacle / latency fields (custom sims may provide these)
    # When unset, they default to 0.0 and do not affect legacy loss modes.
    t_det_s: float = 0.0
    t_det_s_p95: float = 0.0
    episode_time_s: float = 0.0
    safety_success: float = 0.0
    # Fraction of rollouts that ended before every obstacle was first seen in FOV+range
    # (e.g. collision/timeout; MuJoCo inner loop).
    detection_miss_rate: float = 0.0
    # MuJoCo vehicle kinematic / trial logging (optional)
    coverage_fraction: float = 0.0
    n_detected: float = 0.0
    n_obstacles: float = 0.0
    mean_detection_distance_m: float = 0.0
    first_detection_time_mean: float = 0.0
    per_slot_first_hits: dict | None = None


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
    ttw = loss_cfg.get("trial_weight_overrides")
    if ttw is not None and isinstance(ttw, dict):
        w["trial_weights"] = ttw
    return w


def compute_loss(
    metrics: EvalMetrics,
    config: SensorConfig,
    sensor_models: dict,
    weights: dict,
    max_cost_usd: float = 10_000.0,
    hardware_constraints: dict | None = None,
    loss_mode: str = "default",
    experiment_config: dict | None = None,
    loss_config: dict | None = None,
) -> LossResult:
    if loss_mode in ("trial_accuracy", "trial_speed", "trial_multi_objective"):
        from sensor_opt.objectives.trial_objectives import compute_trial_loss
        from sensor_opt.simulation.sensor_specs import get_sensor_specs

        trial_type = "accuracy" if loss_mode == "trial_accuracy" else (
            "speed" if loss_mode == "trial_speed" else "multi_objective"
        )
        tw = (weights or {}).get("trial_weights")
        if tw is None and loss_config and isinstance(loss_config.get("trial_weight_overrides"), dict):
            tw = loss_config.get("trial_weight_overrides")
        if tw is not None and not isinstance(tw, dict):
            tw = None
        max_hw = float(max_cost_usd)
        if loss_config and loss_config.get("max_hardware_budget_usd") is not None:
            max_hw = float(loss_config["max_hardware_budget_usd"])
        elif experiment_config and experiment_config.get("max_hardware_budget_usd") is not None:
            max_hw = float(experiment_config["max_hardware_budget_usd"])
        sspec = get_sensor_specs(experiment_config)
        return compute_trial_loss(
            trial_type=trial_type,
            metrics=metrics,
            config=config,
            sensor_models=sensor_models,
            trial_weights=tw,
            max_hardware_budget_usd=max_hw,
            sensor_specs=sspec,
        )

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
    elif loss_mode == "mujoco_tri":
        # Speed (alpha): p95 time until last obstacle is first detected, vs horizon.
        # Accuracy (beta): share of episodes that failed to see all obstacles before end.
        # Cost (gamma): sensor budget, plus hardware penalty below.
        p95n = float(xp.clip(float(getattr(metrics, "t_det_s_p95", 0.0)) / max(t_max, 1e-6), 0.0, 1.0))
        dmr = float(xp.clip(float(getattr(metrics, "detection_miss_rate", 0.0)), 0.0, 1.0))
        blind_term = float(alpha * p95n)
        collision_term = float(beta * dmr)
        cost_usd = _compute_effective_cost(config, sensor_models)
        cost_penalty = float(xp.clip(cost_usd / max_cost_usd, 0.0, 1.0))
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
        elif loss_mode == "mujoco_tri":
            # Worst p95 and worst miss; $ and hardware terms stay zero
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
    elif loss_mode == "mujoco_tri":
        dmr = float(_clamp(getattr(metrics, "detection_miss_rate", 0.0)))
        p95n = float(_clamp(float(getattr(metrics, "t_det_s_p95", 0.0)) / max(t_max, 1e-6)))
        objectives = {
            "collision": dmr,
            "blind_spot": p95n,
            "cost": cost_penalty,
            "hardware": float(_clamp(hardware_penalty_term)),
            "t_det_s_p95": float(getattr(metrics, "t_det_s_p95", 0.0)),
            "detection_miss_rate": dmr,
        }
    else:
        objectives = {
            "collision": float(_clamp(metrics.collision_rate)),
            "blind_spot": float(_clamp(metrics.blind_spot_fraction)),
            "cost": cost_penalty,
            "hardware": float(_clamp(hardware_penalty_term)),
        }

    total_out = float(total) if loss_mode in ("obstacle_latency", "mujoco_tri") else float(_clamp(total))

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
    """JAX numpy API for clipping / bounds in loss-side helpers."""
    return jnp


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