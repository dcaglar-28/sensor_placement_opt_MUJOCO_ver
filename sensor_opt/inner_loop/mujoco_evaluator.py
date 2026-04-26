"""
MuJoCo inner loop: reuses the generic batched sim evaluator’s interface.

Wire a `MujocoEnvManager` (constructed from YAML via `run_experiment` and
`inner_loop.mujoco`, or pass `env=` for tests).
"""

from __future__ import annotations

from sensor_opt.inner_loop.isaac_evaluator import IsaacSimEvaluator
from sensor_opt.inner_loop.mujoco_env_manager import MujocoEnvManager


class MujocoSimEvaluator(IsaacSimEvaluator):
    """
    Batched rollouts with a `MujocoEnvManager` as `env`, unless one is passed
    explicitly (e.g. tests).
    """

    def __init__(self, mujoco_cfg: dict | None = None) -> None:
        m = dict(mujoco_cfg or {})
        env = m.pop("env", None)
        num_envs = int(m.pop("num_envs", 1))
        sn = float(m.pop("sensor_noise_std", 0.0) or 0.0)
        mstep = int(m.pop("max_steps_per_episode", 500))
        if env is None:
            env = MujocoEnvManager(
                num_envs=num_envs,
                max_steps_per_episode=mstep,
                _sensor_noise_std=sn,
                **m,
            )
        else:
            num_envs = int(getattr(env, "num_envs", num_envs))
        super().__init__(isaac_sim_cfg={"env": env, "num_envs": num_envs, "sensor_noise_std": sn})
