"""MuJoCo inner loop (install `mujoco` from `requirements.txt`)."""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

import numpy as np

from sensor_opt.encoding.config import SensorConfig, SingleSensorConfig
from sensor_opt.inner_loop.mujoco_env_manager import MujocoEnvManager
from sensor_opt.inner_loop.mujoco_evaluator import MujocoSimEvaluator


def test_mujoco_env_manager_returns_metrics_per_env() -> None:
    m = MujocoEnvManager(num_envs=2, max_steps_per_episode=80)
    m.reconfigure_sensors(0, SensorConfig(sensors=[]), {})
    m.reconfigure_sensors(
        1,
        SensorConfig(
            sensors=[
                SingleSensorConfig(
                    sensor_type="lidar",
                    slot="front",
                    range_fraction=1.0,
                    hfov_fraction=1.0,
                )
            ]
        ),
        {"lidar": {"range_m": 100.0, "horizontal_fov_deg": 360.0}},
    )
    out = m.run_rollouts(4, np.random.default_rng(1), sensor_noise_std=0.0)
    assert len(out) == 2
    assert all(0.0 <= x.collision_rate <= 1.0 for x in out)
    assert all(0.0 <= x.blind_spot_fraction <= 1.0 for x in out)


def test_mujoco_sim_evaluator_runbatch() -> None:
    ev = MujocoSimEvaluator(
        mujoco_cfg={"num_envs": 1, "max_steps_per_episode": 60, "sensor_noise_std": 0.0}
    )
    cfg = SensorConfig(
        sensors=[SingleSensorConfig(sensor_type="camera", slot="front", range_fraction=1.0, hfov_fraction=1.0)]
    )
    sm = {"camera": {"range_m": 15.0, "horizontal_fov_deg": 87.0}}
    m = ev.run(cfg, sm, n_episodes=3, rng=np.random.default_rng(2))
    assert m.n_episodes == 3
