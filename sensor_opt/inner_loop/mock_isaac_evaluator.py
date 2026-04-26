"""
Mock high-fidelity evaluator that mimics slow physics-style rollouts (latency + noise).
"""

from __future__ import annotations

import time

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.evaluation.base import BaseEvaluator
from sensor_opt.inner_loop.baseline_metrics import fast_baseline_metrics
from sensor_opt.loss.loss import EvalMetrics


class MockIsaacEvaluator(BaseEvaluator):
    """Slow, stochastic evaluator placeholder for a heavy sim."""

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

        base = fast_baseline_metrics(
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
        generation: int = 0,
    ) -> list[EvalMetrics]:
        _ = generation
        # Mock "batched" mode: still sequential, but matches the interface that
        # a real vectorized backend would implement.
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
    extra high-fidelity-like variability.
    """
    evaluator = MockIsaacEvaluator(
        latency_sec=latency_sec,
        stochastic_std=stochastic_std,
        baseline_noise_std=float(noise_std),
    )
    return evaluator.run(config=config, sensor_models=sensor_models, n_episodes=n_episodes, rng=rng)


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))
