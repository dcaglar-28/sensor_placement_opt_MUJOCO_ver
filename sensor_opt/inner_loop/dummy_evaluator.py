"""
sensor_opt/inner_loop/dummy_evaluator.py

Structured dummy evaluator — gives CMA-ES a real gradient so you can
see convergence during Phase 0 without Isaac Sim.

Oracle score is based on:
  - Slot position quality (top=1.0, sides=0.55)
  - Sensor type effectiveness (LiDAR > radar > camera)
  - z-height bonus
  - Range and FOV utilisation
"""

from __future__ import annotations

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.evaluation.base import BaseEvaluator
from sensor_opt.loss.loss import EvalMetrics

SLOT_COVERAGE_SCORE = {
    "top":         1.0,
    "front":       0.80,
    "rear":        0.75,
    "front-left":  0.70,
    "front-right": 0.70,
    "rear-left":   0.65,
    "rear-right":  0.65,
    "left":        0.55,
    "right":       0.55,
}

TYPE_COLLISION_WEIGHT = {
    "lidar":  0.90,
    "camera": 0.55,
    "radar":  0.70,
}
TYPE_BLIND_WEIGHT = {
    "lidar":  0.85,
    "camera": 0.50,
    "radar":  0.60,
}


class FastEvaluator(BaseEvaluator):
    """
    Fast analytical evaluator (Phase 0 / low-fidelity).
    Reuses the original dummy evaluation logic.
    """

    def __init__(self, noise_std: float = 0.05):
        self.noise_std = noise_std

    def run(
        self,
        config: SensorConfig,
        sensor_models: dict,
        n_episodes: int = 15,
        rng: np.random.Generator | None = None,
    ) -> EvalMetrics:
        return _evaluate_core(
            config=config,
            sensor_models=sensor_models,
            n_episodes=n_episodes,
            noise_std=self.noise_std,
            rng=rng,
        )


def evaluate(
    config: SensorConfig,
    sensor_models: dict,
    n_episodes: int = 15,
    noise_std: float = 0.05,
    rng: np.random.Generator | None = None,
) -> EvalMetrics:
    """Backwards-compatible function API used by existing tests/callers."""
    return _evaluate_core(
        config=config,
        sensor_models=sensor_models,
        n_episodes=n_episodes,
        noise_std=noise_std,
        rng=rng,
    )


def _evaluate_core(
    config: SensorConfig,
    sensor_models: dict,
    n_episodes: int,
    noise_std: float,
    rng: np.random.Generator | None,
) -> EvalMetrics:
    if rng is None:
        rng = np.random.default_rng()

    active = config.active_sensors()

    if not active:
        return EvalMetrics(
            collision_rate=1.0,
            blind_spot_fraction=1.0,
            mean_goal_success=0.0,
            n_episodes=n_episodes,
        )

    collision_coverage = 0.0
    blind_coverage = 0.0

    for sensor in active:
        cw = TYPE_COLLISION_WEIGHT.get(sensor.sensor_type, 0.5)
        bw = TYPE_BLIND_WEIGHT.get(sensor.sensor_type, 0.5)
        slot_score = SLOT_COVERAGE_SCORE.get(sensor.slot, 0.5)
        model = sensor_models.get(sensor.sensor_type, {})

        z_bonus   = min(sensor.z_offset / 0.5, 1.0) * 0.15
        range_bonus = sensor.range_fraction * 0.1
        fov_h     = float(model.get("horizontal_fov_deg", 90.0))
        fov_bonus = sensor.hfov_fraction * (fov_h / 360.0) * 0.2

        collision_coverage += cw * (slot_score + z_bonus + range_bonus)
        blind_coverage     += bw * (slot_score + z_bonus + fov_bonus)

    n = len(active)
    norm_factor = 1.0 - np.exp(-n / 2.0)

    col_norm   = _clamp(collision_coverage / (n + 1e-6) * norm_factor)
    blind_norm = _clamp(blind_coverage     / (n + 1e-6) * norm_factor)

    base_collision_rate = _clamp(1.0 - col_norm)
    base_blind_frac     = _clamp(1.0 - blind_norm)
    base_success        = col_norm * 0.8

    episode_collisions = 0
    episode_successes  = 0

    for _ in range(n_episodes):
        ep_noise = rng.normal(0.0, noise_std)
        ep_col_prob = _clamp(base_collision_rate + ep_noise)
        had_collision = rng.random() < ep_col_prob
        ep_success_prob = _clamp(base_success - ep_noise * 0.5)
        reached_goal = (not had_collision) and (rng.random() < ep_success_prob)
        episode_collisions += int(had_collision)
        episode_successes  += int(reached_goal)

    blind_frac = _clamp(base_blind_frac + rng.normal(0.0, noise_std * 0.5))

    return EvalMetrics(
        collision_rate=episode_collisions / n_episodes,
        blind_spot_fraction=blind_frac,
        mean_goal_success=episode_successes / n_episodes,
        n_episodes=n_episodes,
    )


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))