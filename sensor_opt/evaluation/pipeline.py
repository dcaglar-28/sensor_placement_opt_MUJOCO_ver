"""
Multi-fidelity evaluation orchestration.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from sensor_opt.design.config import DesignConfig, build_design_config
from sensor_opt.encoding.config import SensorConfig
from sensor_opt.evaluation.base import BaseEvaluator
from sensor_opt.evaluation.results import EvaluationResult
from sensor_opt.loss.loss import EvalMetrics, compute_loss


@dataclass
class Evaluator:
    """Three-stage evaluator: fast -> mid -> slow."""

    fast_eval: BaseEvaluator
    mid_eval: BaseEvaluator
    slow_eval: BaseEvaluator
    weights: dict
    sensor_models: dict
    max_cost_usd: float = 10_000.0
    fast_collision_threshold: float = 0.95
    promising_collision_threshold: float = 0.35

    def evaluate(
        self,
        config: SensorConfig,
        n_episodes: int = 15,
        rng: Optional[np.random.Generator] = None,
        cfg: Optional[dict] = None,
    ) -> EvaluationResult:
        if rng is None:
            rng = np.random.default_rng()

        start = time.perf_counter()
        design = build_design_config(config, cfg)

        fast = self.fast_eval.run(
            config=config,
            sensor_models=self.sensor_models,
            n_episodes=max(3, n_episodes // 3),
            rng=rng,
        )

        if fast.collision_rate > self.fast_collision_threshold:
            return self._finalize_result(fast, "fast_reject", design, time.perf_counter() - start)

        mid = self.mid_eval.run(
            config=config,
            sensor_models=self.sensor_models,
            n_episodes=max(5, (2 * n_episodes) // 3),
            rng=rng,
        )

        promising = mid.collision_rate <= self.promising_collision_threshold
        final_metrics = self.slow_eval.run(
            config=config,
            sensor_models=self.sensor_models,
            n_episodes=n_episodes,
            rng=rng,
        ) if promising else mid
        fidelity = "slow" if promising else "mid"
        return self._finalize_result(final_metrics, fidelity, design, time.perf_counter() - start)

    def _finalize_result(
        self,
        metrics: EvalMetrics,
        fidelity: str,
        design: DesignConfig,
        elapsed_sec: float,
    ) -> EvaluationResult:
        loss = compute_loss(
            metrics=metrics,
            config=design.sensors,
            sensor_models=self.sensor_models,
            weights=self.weights,
            max_cost_usd=self.max_cost_usd,
            hardware_constraints={
                "compute_limit_tops": design.hardware.compute_limit_tops,
                "memory_limit_gb": design.hardware.memory_limit_gb,
                "latency_budget_ms": design.hardware.latency_budget_ms,
            },
        )
        return EvaluationResult(
            metrics=metrics,
            loss=loss,
            objectives=dict(loss.objectives),
            fidelity=fidelity,
            evaluation_time_sec=elapsed_sec,
        )
