"""
Mock high-fidelity evaluator that mimics slow Isaac Sim behavior.
"""

from __future__ import annotations

import time

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


def _fast_baseline_metrics(
    config: SensorConfig,
    sensor_models: dict,
    n_episodes: int,
    rng: np.random.Generator,
    noise_std: float,
) -> EvalMetrics:
    """
    Lightweight analytic baseline so `MockIsaacEvaluator` is self-contained.

    This used to live in `dummy_evaluator`; we keep the same general behavior
    (slot/type/height/range/fov heuristics + episode noise) but avoid importing
    the dummy module so the repo has a single non-Isaac evaluator.
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
    col_norm = _clamp(collision_coverage / (n + 1e-6) * norm_factor)
    blind_norm = _clamp(blind_coverage / (n + 1e-6) * norm_factor)

    base_collision_rate = _clamp(1.0 - col_norm)
    base_blind_frac = _clamp(1.0 - blind_norm)
    base_success = col_norm * 0.8

    episode_collisions = 0
    episode_successes = 0
    for _ in range(n_episodes):
        ep_noise = rng.normal(0.0, noise_std)
        ep_col_prob = _clamp(base_collision_rate + ep_noise)
        had_collision = rng.random() < ep_col_prob
        ep_success_prob = _clamp(base_success - ep_noise * 0.5)
        reached_goal = (not had_collision) and (rng.random() < ep_success_prob)
        episode_collisions += int(had_collision)
        episode_successes += int(reached_goal)

    blind_frac = _clamp(base_blind_frac + rng.normal(0.0, noise_std * 0.5))
    return EvalMetrics(
        collision_rate=episode_collisions / n_episodes,
        blind_spot_fraction=blind_frac,
        mean_goal_success=episode_successes / n_episodes,
        n_episodes=n_episodes,
    )


class MockIsaacEvaluator(BaseEvaluator):
    """Slow, stochastic evaluator placeholder for Isaac Sim."""

    def __init__(
        self,
        latency_sec: float = 0.15,
        stochastic_std: float = 0.03,
        baseline_noise_std: float = 0.01,
    ):
        self.latency_sec = latency_sec
        self.stochastic_std = stochastic_std
        self.baseline_noise_std = baseline_noise_std

    def run(
        self,
        config: SensorConfig,
        sensor_models: dict,
        n_episodes: int = 15,
        rng: np.random.Generator | None = None,
    ) -> EvalMetrics:
        if rng is None:
            rng = np.random.default_rng()

        # Mimic simulator wall-clock delay.
        time.sleep(max(0.0, self.latency_sec))

        base = _fast_baseline_metrics(
            config=config,
            sensor_models=sensor_models,
            n_episodes=n_episodes,
            rng=rng,
            noise_std=self.baseline_noise_std,
        )

        # Add extra variance to mimic scenario stochasticity in physics-based sim.
        coll = _clamp(base.collision_rate + rng.normal(0.0, self.stochastic_std))
        blind = _clamp(base.blind_spot_fraction + rng.normal(0.0, self.stochastic_std))
        success = _clamp(base.mean_goal_success + rng.normal(0.0, self.stochastic_std * 0.7))
        return EvalMetrics(
            collision_rate=coll,
            blind_spot_fraction=blind,
            mean_goal_success=success,
            n_episodes=n_episodes,
        )

    def run_batch(
        self,
        configs: list[SensorConfig],
        sensor_models: dict,
        n_episodes: int = 15,
        rng: np.random.Generator | None = None,
    ) -> list[EvalMetrics]:
        # Mock "batched" mode: still sequential, but matches the interface that
        # a real Isaac Sim backend would implement (e.g., vectorized envs).
        return [
            self.run(config=c, sensor_models=sensor_models, n_episodes=n_episodes, rng=rng)
            for c in configs
        ]


def evaluate(
    config: SensorConfig,
    sensor_models: dict,
    n_episodes: int = 15,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    latency_sec: float = 0.15,
    stochastic_std: float = 0.03,
) -> EvalMetrics:
    """
    Backwards-compatible function API (mirrors the old dummy evaluator shape).
    `noise_std` maps to the baseline analytic noise; stochastic_std adds extra
    Isaac-like variability.
    """
    evaluator = MockIsaacEvaluator(
        latency_sec=latency_sec,
        stochastic_std=stochastic_std,
        baseline_noise_std=float(noise_std),
    )
    return evaluator.run(config=config, sensor_models=sensor_models, n_episodes=n_episodes, rng=rng)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))
