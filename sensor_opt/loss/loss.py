"""
sensor_opt/loss/loss.py

Loss function: L = α·collision_rate + β·blind_spot_fraction + γ·cost_penalty

All three terms are in [0, 1].
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


def compute_loss(
    metrics: EvalMetrics,
    config: SensorConfig,
    sensor_models: dict,
    weights: dict,
    max_cost_usd: float = 10_000.0,
    hardware_constraints: dict | None = None,
) -> LossResult:
    xp = _array_lib()

    alpha = weights["alpha"]
    beta  = weights["beta"]
    gamma = weights["gamma"]

    collision_term = float(alpha * xp.clip(float(metrics.collision_rate), 0.0, 1.0))
    blind_term     = float(beta  * xp.clip(float(metrics.blind_spot_fraction), 0.0, 1.0))

    cost_usd    = _compute_effective_cost(config, sensor_models)
    cost_penalty = float(xp.clip(cost_usd / max_cost_usd, 0.0, 1.0))
    cost_term   = float(gamma * cost_penalty)

    hardware_penalty_term = _compute_hardware_penalty(config, sensor_models, hardware_constraints)
    total = collision_term + blind_term + cost_term + hardware_penalty_term

    if not config.active_sensors():
        total = 1.0

    objectives = {
        "collision": float(_clamp(metrics.collision_rate)),
        "blind_spot": float(_clamp(metrics.blind_spot_fraction)),
        "cost": cost_penalty,
        "hardware": float(_clamp(hardware_penalty_term)),
    }

    return LossResult(
        total=_clamp(total),
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