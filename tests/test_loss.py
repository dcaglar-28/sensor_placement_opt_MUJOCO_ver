"""tests/test_loss.py"""

import pytest
from sensor_opt.encoding.config import SingleSensorConfig, SensorConfig
from sensor_opt.loss.loss import EvalMetrics, LossResult, compute_loss

SENSOR_MODELS = {
    "lidar":  {"cost_usd": 4000, "horizontal_fov_deg": 360.0},
    "camera": {"cost_usd": 200,  "horizontal_fov_deg": 87.0},
    "radar":  {"cost_usd": 650,  "horizontal_fov_deg": 120.0},
}
WEIGHTS = {"alpha": 0.4, "beta": 0.4, "gamma": 0.2}
MAX_COST = 10_000.0


def make_metrics(col=0.1, blind=0.2, success=0.8, n=15):
    return EvalMetrics(collision_rate=col, blind_spot_fraction=blind,
                       mean_goal_success=success, n_episodes=n)

def make_config(*types):
    return SensorConfig(sensors=[SingleSensorConfig(t, "top") for t in types])


def test_loss_returns_loss_result():
    assert isinstance(compute_loss(make_metrics(), make_config("lidar"), SENSOR_MODELS, WEIGHTS, MAX_COST), LossResult)

def test_loss_total_in_unit_interval():
    for col in [0.0, 0.5, 1.0]:
        for blind in [0.0, 0.5, 1.0]:
            r = compute_loss(make_metrics(col=col, blind=blind), make_config("lidar"), SENSOR_MODELS, WEIGHTS, MAX_COST)
            assert 0.0 <= r.total <= 1.0

def test_loss_terms_sum_to_total_approximately():
    r = compute_loss(make_metrics(0.2, 0.3), make_config("lidar"), SENSOR_MODELS, WEIGHTS, MAX_COST)
    assert abs(r.collision_term + r.blind_term + r.cost_term - r.total) < 1e-6

def test_higher_collision_rate_increases_loss():
    cfg = make_config("lidar")
    assert compute_loss(make_metrics(col=0.9), cfg, SENSOR_MODELS, WEIGHTS, MAX_COST).total > \
           compute_loss(make_metrics(col=0.1), cfg, SENSOR_MODELS, WEIGHTS, MAX_COST).total

def test_higher_blind_spot_increases_loss():
    cfg = make_config("lidar")
    assert compute_loss(make_metrics(blind=0.9), cfg, SENSOR_MODELS, WEIGHTS, MAX_COST).total > \
           compute_loss(make_metrics(blind=0.1), cfg, SENSOR_MODELS, WEIGHTS, MAX_COST).total

def test_more_expensive_config_has_higher_cost_term():
    r_cheap = compute_loss(make_metrics(), make_config("camera"),        SENSOR_MODELS, WEIGHTS, MAX_COST)
    r_exp   = compute_loss(make_metrics(), make_config("lidar", "lidar"), SENSOR_MODELS, WEIGHTS, MAX_COST)
    assert r_exp.cost_term > r_cheap.cost_term

def test_empty_config_returns_max_loss():
    assert compute_loss(make_metrics(), SensorConfig(sensors=[]), SENSOR_MODELS, WEIGHTS, MAX_COST).total == 1.0

def test_all_disabled_returns_max_loss():
    assert compute_loss(make_metrics(), make_config("disabled", "disabled"), SENSOR_MODELS, WEIGHTS, MAX_COST).total == 1.0

def test_perfect_metrics_gives_low_loss():
    r = compute_loss(make_metrics(col=0.0, blind=0.0), make_config("lidar"), SENSOR_MODELS, WEIGHTS, MAX_COST)
    assert r.total < 0.15

def test_worst_metrics_gives_high_loss():
    r = compute_loss(make_metrics(col=1.0, blind=1.0), make_config("lidar"), SENSOR_MODELS, WEIGHTS, MAX_COST)
    assert r.total > 0.8

def test_cost_usd_is_positive_for_active_sensor():
    assert compute_loss(make_metrics(), make_config("lidar"), SENSOR_MODELS, WEIGHTS, MAX_COST).cost_usd > 0.0

def test_cost_usd_zero_for_disabled():
    assert compute_loss(make_metrics(), make_config("disabled"), SENSOR_MODELS, WEIGHTS, MAX_COST).cost_usd == 0.0

def test_n_active_sensors_correct():
    cfg = SensorConfig(sensors=[
        SingleSensorConfig("lidar", "top"),
        SingleSensorConfig("disabled", "front"),
        SingleSensorConfig("camera", "rear"),
    ])
    assert compute_loss(make_metrics(), cfg, SENSOR_MODELS, WEIGHTS, MAX_COST).n_active_sensors == 2

def test_alpha_zero_collision_ignored():
    r = compute_loss(make_metrics(col=1.0, blind=0.0), make_config("lidar"),
                     SENSOR_MODELS, {"alpha": 0.0, "beta": 0.5, "gamma": 0.5}, MAX_COST)
    assert r.collision_term == 0.0

def test_gamma_zero_cost_ignored():
    r = compute_loss(make_metrics(), make_config("lidar"),
                     SENSOR_MODELS, {"alpha": 0.5, "beta": 0.5, "gamma": 0.0}, MAX_COST)
    assert r.cost_term == 0.0


def test_obstacle_latency_mode_matches_expected_weighting():
    w = {"alpha": 1.0, "beta": 100.0, "gamma": 0.0, "t_det_max_s": 10.0}
    m = EvalMetrics(
        collision_rate=0.1,
        blind_spot_fraction=0.0,
        mean_goal_success=0.0,
        n_episodes=10,
        t_det_s=0.0,
        t_det_s_p95=5.0,
        episode_time_s=10.0,
        safety_success=0.0,
    )
    r = compute_loss(
        m,
        make_config("lidar"),
        SENSOR_MODELS,
        w,
        MAX_COST,
        hardware_constraints={},
        loss_mode="obstacle_latency",
    )
    assert abs(r.blind_term - 0.5) < 1e-6  # alpha * (5/10)
    assert abs(r.collision_term - 10.0) < 1e-6  # beta * 0.1
    assert abs(r.total - 10.5) < 1e-6