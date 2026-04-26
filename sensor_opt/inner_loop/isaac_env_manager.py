"""
Environment manager protocol + a concrete mock for vectorized inner-loop tests.

`run_batch()` on the sim evaluator base expects an object with:
  - reconfigure_sensors(env_idx, config, sensor_models)
  - run_rollouts(n_episodes, rng) -> list[EvalMetrics]

This module provides:
  - `IsaacEnvManagerProtocol`: a structural typing helper (optional)
  - `MockIsaacEnvManager`: a concrete implementation you can run today

To integrate a real backend, implement the same two methods and drive your
scene/sensor setup + vectorized stepping there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.inner_loop.baseline_metrics import fast_baseline_metrics
from sensor_opt.loss.loss import EvalMetrics


class IsaacEnvManagerProtocol(Protocol):
    def reconfigure_sensors(self, env_idx: int, config: SensorConfig, sensor_models: dict) -> None: ...

    def run_rollouts(self, n_episodes: int, rng: np.random.Generator) -> list[EvalMetrics]: ...


@dataclass
class _SlotState:
    config: Optional[SensorConfig] = None
    sensor_models: Optional[dict] = None


class MockIsaacEnvManager:
    """
    Concrete env manager that mimics a vectorized multi-env setup.

    - Holds `num_envs` independent "slots" (like parallel environments).
    - `reconfigure_sensors` just stores the config per slot (no full reset).
    - `run_rollouts` evaluates all slots and returns one EvalMetrics per slot.

    This validates ordering/chunking/batching end-to-end before wiring a real
    physics or game engine backend.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        baseline_noise_std: float = 0.01,
        stochastic_std: float = 0.03,
    ):
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.baseline_noise_std = float(baseline_noise_std)
        self.stochastic_std = float(stochastic_std)
        self._slots = [_SlotState() for _ in range(self.num_envs)]

    def reconfigure_sensors(self, env_idx: int, config: SensorConfig, sensor_models: dict) -> None:
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(f"env_idx out of range: {env_idx}")
        self._slots[env_idx].config = config
        self._slots[env_idx].sensor_models = sensor_models

    def run_rollouts(self, n_episodes: int, rng: np.random.Generator) -> list[EvalMetrics]:
        out: list[EvalMetrics] = []

        # In a real backend, this is where you'd step all envs
        # together each sim tick. Here we just compute per-slot metrics.
        for env_idx in range(self.num_envs):
            slot = self._slots[env_idx]
            cfg = slot.config or SensorConfig(sensors=[])
            models = slot.sensor_models or {}

            base = fast_baseline_metrics(
                config=cfg,
                sensor_models=models,
                n_episodes=n_episodes,
                rng=rng,
                noise_std=self.baseline_noise_std,
            )

            # Add extra variance to mimic high-fidelity sim stochasticity.
            coll = float(np.clip(base.collision_rate + rng.normal(0.0, self.stochastic_std), 0.0, 1.0))
            blind = float(np.clip(base.blind_spot_fraction + rng.normal(0.0, self.stochastic_std), 0.0, 1.0))
            success = float(np.clip(base.mean_goal_success + rng.normal(0.0, self.stochastic_std * 0.7), 0.0, 1.0))
            out.append(
                EvalMetrics(
                    collision_rate=coll,
                    blind_spot_fraction=blind,
                    mean_goal_success=success,
                    n_episodes=n_episodes,
                )
            )

        return out

