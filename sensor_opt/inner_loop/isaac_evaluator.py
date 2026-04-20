"""
sensor_opt/inner_loop/isaac_evaluator.py

Isaac Sim inner-loop stub. Implement Phase 1 here.
Public interface matches dummy_evaluator.evaluate() exactly.
"""

from __future__ import annotations

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.evaluation.base import BaseEvaluator
from sensor_opt.loss.loss import EvalMetrics


class IsaacSimEvaluator(BaseEvaluator):
    """
    Isaac Sim integration layer placeholder.

    Expected inputs:
      - config: SensorConfig with sensor type, slot, 3D position offsets, orientation,
                range fraction and FoV fraction.
      - sensor_models: dict with per-type metadata (cost, FoV, sensor characteristics).
      - n_episodes: number of evaluation rollouts to execute in simulator.
      - rng: optional numpy generator for deterministic sampling or scenario randomization.

    Expected output:
      - EvalMetrics(collision_rate, blind_spot_fraction, mean_goal_success, n_episodes)
    """

    def __init__(self, isaac_sim_cfg: dict | None = None):
        self.isaac_sim_cfg = isaac_sim_cfg or {}

    def run(
        self,
        config: SensorConfig,
        sensor_models: dict,
        n_episodes: int = 15,
        rng: np.random.Generator | None = None,
    ) -> EvalMetrics:
        raise NotImplementedError(
            "Isaac Sim evaluator not yet implemented. "
            "Run with --dummy flag or use mock_isaac mode."
        )


def evaluate(
    config: SensorConfig,
    sensor_models: dict,
    n_episodes: int = 15,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    isaac_sim_cfg: dict | None = None,
) -> EvalMetrics:
    """Backwards-compatible function API."""
    _ = noise_std
    evaluator = IsaacSimEvaluator(isaac_sim_cfg=isaac_sim_cfg)
    return evaluator.run(config=config, sensor_models=sensor_models, n_episodes=n_episodes, rng=rng)