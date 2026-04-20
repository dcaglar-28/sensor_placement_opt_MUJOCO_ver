"""
Mock high-fidelity evaluator that mimics slow Isaac Sim behavior.
"""

from __future__ import annotations

import time

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.evaluation.base import BaseEvaluator
from sensor_opt.inner_loop.dummy_evaluator import evaluate as fast_evaluate
from sensor_opt.loss.loss import EvalMetrics


class MockIsaacEvaluator(BaseEvaluator):
    """Slow, stochastic evaluator placeholder for Isaac Sim."""

    def __init__(self, latency_sec: float = 0.15, stochastic_std: float = 0.03):
        self.latency_sec = latency_sec
        self.stochastic_std = stochastic_std

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

        base = fast_evaluate(
            config=config,
            sensor_models=sensor_models,
            n_episodes=n_episodes,
            noise_std=0.01,
            rng=rng,
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


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(v)))
