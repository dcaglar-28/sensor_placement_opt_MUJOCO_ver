"""tests/test_multi_fidelity.py"""

import numpy as np

from sensor_opt.encoding.config import SensorConfig, SingleSensorConfig
from sensor_opt.evaluation.pipeline import Evaluator
from sensor_opt.inner_loop.mock_isaac_evaluator import MockIsaacEvaluator


SENSOR_MODELS = {
    "lidar": {"cost_usd": 4000, "horizontal_fov_deg": 360.0, "compute_tops": 8.0, "memory_gb": 1.2, "latency_ms": 20.0},
    "camera": {"cost_usd": 200, "horizontal_fov_deg": 87.0, "compute_tops": 2.0, "memory_gb": 0.4, "latency_ms": 8.0},
}


def test_multi_fidelity_returns_objectives():
    cfg = SensorConfig(sensors=[SingleSensorConfig("lidar", "top", z_offset=0.3)])
    evaluator = Evaluator(
        fast_eval=MockIsaacEvaluator(latency_sec=0.0, stochastic_std=0.01, baseline_noise_std=0.05),
        mid_eval=MockIsaacEvaluator(latency_sec=0.0, stochastic_std=0.008, baseline_noise_std=0.03),
        slow_eval=MockIsaacEvaluator(latency_sec=0.0, stochastic_std=0.01, baseline_noise_std=0.01),
        weights={"alpha": 0.4, "beta": 0.4, "gamma": 0.2},
        sensor_models=SENSOR_MODELS,
        max_cost_usd=10_000.0,
    )
    result = evaluator.evaluate(
        config=cfg,
        n_episodes=8,
        rng=np.random.default_rng(7),
        cfg={"hardware": {"compute_limit_tops": 20.0, "memory_limit_gb": 4.0, "latency_budget_ms": 60.0}},
    )
    assert "collision" in result.objectives
    assert "blind_spot" in result.objectives
    assert "cost" in result.objectives
