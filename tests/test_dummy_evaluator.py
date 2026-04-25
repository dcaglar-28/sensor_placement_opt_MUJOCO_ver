"""tests/test_dummy_evaluator.py

Legacy name: this now tests the self-contained MockIsaac evaluator baseline.
"""

import numpy as np
import pytest

from sensor_opt.encoding.config import SingleSensorConfig, SensorConfig
from sensor_opt.inner_loop.mock_isaac_evaluator import evaluate
from sensor_opt.loss.loss import EvalMetrics

SENSOR_MODELS = {
    "lidar":  {"cost_usd": 4000, "horizontal_fov_deg": 360.0, "blind_spot_reduction_factor": 0.85},
    "camera": {"cost_usd": 200,  "horizontal_fov_deg": 87.0,  "blind_spot_reduction_factor": 0.45},
    "radar":  {"cost_usd": 650,  "horizontal_fov_deg": 120.0, "blind_spot_reduction_factor": 0.60},
}
RNG = np.random.default_rng(42)


def make_cfg(*types_slots):
    return SensorConfig(sensors=[SingleSensorConfig(t, s, z_offset=0.3) for t, s in types_slots])


def test_returns_eval_metrics():
    assert isinstance(evaluate(make_cfg(("lidar", "top")), SENSOR_MODELS, n_episodes=5, rng=RNG), EvalMetrics)

def test_metrics_in_unit_interval():
    r = evaluate(make_cfg(("lidar", "top"), ("camera", "front")), SENSOR_MODELS, n_episodes=10, rng=RNG)
    assert 0.0 <= r.collision_rate      <= 1.0
    assert 0.0 <= r.blind_spot_fraction <= 1.0
    assert 0.0 <= r.mean_goal_success   <= 1.0

def test_empty_config_returns_worst():
    r = evaluate(SensorConfig(sensors=[]), SENSOR_MODELS, n_episodes=5, rng=RNG)
    assert r.collision_rate == 1.0 and r.blind_spot_fraction == 1.0

def test_all_disabled_returns_worst():
    r = evaluate(make_cfg(("disabled", "front"), ("disabled", "rear")), SENSOR_MODELS, n_episodes=5, rng=RNG)
    assert r.collision_rate == 1.0 and r.blind_spot_fraction == 1.0

def test_lidar_on_top_beats_empty():
    r_empty = evaluate(SensorConfig(sensors=[]), SENSOR_MODELS, n_episodes=50, rng=np.random.default_rng(0))
    r_good  = evaluate(make_cfg(("lidar", "top")), SENSOR_MODELS, n_episodes=50, rng=np.random.default_rng(0))
    assert r_good.collision_rate < r_empty.collision_rate
    assert r_good.blind_spot_fraction < r_empty.blind_spot_fraction

def test_more_sensors_generally_better():
    r_one = evaluate(make_cfg(("lidar", "top")), SENSOR_MODELS, n_episodes=100, rng=np.random.default_rng(1))
    r_two = evaluate(make_cfg(("lidar", "top"), ("camera", "front")), SENSOR_MODELS, n_episodes=100, rng=np.random.default_rng(1))
    assert r_two.collision_rate <= r_one.collision_rate + 0.15

def test_n_episodes_respected():
    r = evaluate(make_cfg(("lidar", "top")), SENSOR_MODELS, n_episodes=7, rng=RNG)
    assert r.n_episodes == 7

def test_zero_noise_is_deterministic():
    r_a = evaluate(make_cfg(("lidar", "top")), SENSOR_MODELS, n_episodes=10, noise_std=0.0, rng=np.random.default_rng(99))
    r_b = evaluate(make_cfg(("lidar", "top")), SENSOR_MODELS, n_episodes=10, noise_std=0.0, rng=np.random.default_rng(99))
    assert r_a.collision_rate == r_b.collision_rate
    assert r_a.blind_spot_fraction == r_b.blind_spot_fraction

def test_cma_converges_on_dummy(tmp_path):
    from sensor_opt.cma.outer_loop import run_outer_loop
    from sensor_opt.encoding.config import make_initial_vector
    from sensor_opt.inner_loop.mock_isaac_evaluator import evaluate as dummy_eval
    from sensor_opt.logging.experiment_logger import ExperimentLogger

    sensor_budget = {"lidar": {"max_count": 1}, "camera": {"max_count": 1}}
    mounting_slots = ["front", "rear", "top", "left", "right"]
    # All-disabled CMA-ES start + pycma/numpy RNG differs across Python versions, so
    # seed=0 can fail on 3.12 with every decode staying disabled. Start one slot
    # as an active LiDAR (type index 1, active flag 0.6) so the run always sees
    # a loss < 1.0 while still exercising the full loop.
    x0 = make_initial_vector(sensor_budget, mounting_slots)
    x0[0] = 1.0  # lidar (SENSOR_TYPE_MAP: 0=disabled, 1=lidar, …)
    x0[1] = 0.6  # active

    cfg = {
        "experiment": {"name": "smoke_test", "seed": 0},
        "sensor_budget": sensor_budget,
        "mounting_slots": mounting_slots,
        "sensor_models": SENSOR_MODELS,
        # tolx/tolfun=0 disables those stop criteria (0 is falsy in cma) so a single
        # "flat" first generation does not end the run on CI; see outer_loop tie-jitter.
        "cma": {
            "x0": x0.tolist(),  # JSON-serializable for ExperimentLogger
            "sigma0": 0.35,
            "population_size": 10,
            "max_generations": 15,
            "tolx": 0,
            "tolfun": 0,
            "tolfunhist": 0,  # do not treat flat 1.0 best history as converged
        },
        "loss": {"alpha": 0.4, "beta": 0.4, "gamma": 0.2, "max_cost_usd": 10000.0},
        "inner_loop": {"mode": "mock_isaac", "n_episodes": 10, "mock_isaac": {"baseline_noise_std": 0.03, "latency_sec": 0.0, "stochastic_std": 0.01}},
        "logging": {"csv": True, "mlflow": False, "log_every_n_generations": 5, "results_dir": str(tmp_path)},
    }

    with ExperimentLogger(experiment_name="smoke_test", results_dir=str(tmp_path),
                          use_mlflow=False, run_config=cfg) as logger:
        result = run_outer_loop(cfg, dummy_eval, logger, seed=0)

    assert result.best_loss < 1.0
    assert result.best_loss < 0.9
    assert result.best_config is not None