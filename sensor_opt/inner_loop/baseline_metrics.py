"""
Analytic baseline metrics used by mock evaluators / mock environment manager.

This is intentionally pure-Python + NumPy so it can run without a physics engine.
"""

from __future__ import annotations

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.loss.loss import EvalMetrics


SLOT_COVERAGE_SCORE = {
    "top": 1.0,
    "front": 0.80,
    "rear": 0.75,
    "front-left": 0.70,
    "front-right": 0.70,
    "rear-left": 0.65,
    "rear-right": 0.65,
    "left": 0.55,
    "right": 0.55,
}

TYPE_COLLISION_WEIGHT = {
    "lidar": 0.90,
    "camera": 0.55,
    "radar": 0.70,
}
TYPE_BLIND_WEIGHT = {
    "lidar": 0.85,
    "camera": 0.50,
    "radar": 0.60,
}


def clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def fast_baseline_metrics(
    config: SensorConfig,
    sensor_models: dict,
    *,
    n_episodes: int,
    rng: np.random.Generator,
    noise_std: float,
) -> EvalMetrics:
    """
    Deterministic-ish (seeded) heuristic that returns EvalMetrics.
    Used as a stand-in for a physics-based inner loop.
    """
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

        z_bonus = min(sensor.z_offset / 0.5, 1.0) * 0.15
        range_bonus = sensor.range_fraction * 0.1
        fov_h = float(model.get("horizontal_fov_deg", 90.0))
        fov_bonus = sensor.hfov_fraction * (fov_h / 360.0) * 0.2

        collision_coverage += cw * (slot_score + z_bonus + range_bonus)
        blind_coverage += bw * (slot_score + z_bonus + fov_bonus)

    n = len(active)
    norm_factor = 1.0 - np.exp(-n / 2.0)
    col_norm = clamp01(collision_coverage / (n + 1e-6) * norm_factor)
    blind_norm = clamp01(blind_coverage / (n + 1e-6) * norm_factor)

    base_collision_rate = clamp01(1.0 - col_norm)
    base_blind_frac = clamp01(1.0 - blind_norm)
    base_success = col_norm * 0.8

    episode_collisions = 0
    episode_successes = 0
    for _ in range(n_episodes):
        ep_noise = rng.normal(0.0, noise_std)
        ep_col_prob = clamp01(base_collision_rate + ep_noise)
        had_collision = rng.random() < ep_col_prob
        ep_success_prob = clamp01(base_success - ep_noise * 0.5)
        reached_goal = (not had_collision) and (rng.random() < ep_success_prob)
        episode_collisions += int(had_collision)
        episode_successes += int(reached_goal)

    blind_frac = clamp01(base_blind_frac + rng.normal(0.0, noise_std * 0.5))
    return EvalMetrics(
        collision_rate=episode_collisions / n_episodes,
        blind_spot_fraction=blind_frac,
        mean_goal_success=episode_successes / n_episodes,
        n_episodes=n_episodes,
    )

