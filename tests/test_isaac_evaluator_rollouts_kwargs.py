from __future__ import annotations

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.inner_loop.isaac_evaluator import IsaacSimEvaluator
from sensor_opt.loss.loss import EvalMetrics


def test_run_rollouts_passes_sensor_noise_when_supported():
    calls: list[dict] = []

    class Env:
        def reconfigure_sensors(self, env_idx, config, sensor_models):
            pass

        def run_rollouts(self, n_episodes, rng, sensor_noise_std=0.0):
            calls.append({"sensor_noise_std": float(sensor_noise_std)})
            return [
                EvalMetrics(
                    collision_rate=0.0,
                    blind_spot_fraction=0.0,
                    mean_goal_success=0.0,
                    n_episodes=n_episodes,
                )
            ]

    ev = IsaacSimEvaluator(
        isaac_sim_cfg={"env": Env(), "num_envs": 1, "sensor_noise_std": 0.05}
    )
    r = np.random.default_rng(0)
    out = ev.run(SensorConfig(sensors=[]), {}, n_episodes=2, rng=r)
    assert len(calls) == 1
    assert calls[0]["sensor_noise_std"] == 0.05
    assert out.n_episodes == 2


def test_run_rollouts_falls_back_when_not_supported():
    class Env:
        def reconfigure_sensors(self, env_idx, config, sensor_models):
            pass

        def run_rollouts(self, n_episodes, rng):
            return [
                EvalMetrics(
                    collision_rate=0.0,
                    blind_spot_fraction=0.0,
                    mean_goal_success=0.0,
                    n_episodes=n_episodes,
                )
            ]

    ev = IsaacSimEvaluator(
        isaac_sim_cfg={"env": Env(), "num_envs": 1, "sensor_noise_std": 0.05}
    )
    r = np.random.default_rng(0)
    out = ev.run(SensorConfig(sensors=[]), {}, n_episodes=1, rng=r)
    assert out.n_episodes == 1
