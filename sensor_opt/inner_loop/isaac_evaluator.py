"""
sensor_opt/inner_loop/isaac_evaluator.py

Isaac Sim inner-loop stub. Implement Phase 1 here.
Public interface matches the inner-loop `evaluate()` shape used by the repo.
"""

from __future__ import annotations

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.evaluation.base import BaseEvaluator
from sensor_opt.loss.loss import EvalMetrics


def _chunked(seq: list, chunk_size: int):
    """
    Yield (start_index, chunk_list) pairs.

    Kept here (instead of a shared utils module) so the Isaac integration can be
    copied into an Isaac Sim project without extra dependencies.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be >= 1")
    for start in range(0, len(seq), chunk_size):
        yield start, seq[start : start + chunk_size]


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
        # Expected to be provided by the user’s Isaac-side integration.
        # Must support:
        #   - reconfigure_sensors(env_idx, config, sensor_models)
        #   - run_rollouts(n_episodes, rng, **optional kwargs) -> list[EvalMetrics] (len == num_envs)
        #     e.g. `sensor_noise_std` (passed through from `inner_loop.isaac_sim` YAML for the HTTP bridge)
        self.env = self.isaac_sim_cfg.get("env", None)
        self.num_envs = int(self.isaac_sim_cfg.get("num_envs", 1))
        if self.num_envs < 1:
            raise ValueError("isaac_sim_cfg['num_envs'] must be >= 1")
        self._sensor_noise_std = float(self.isaac_sim_cfg.get("sensor_noise_std", 0.0) or 0.0)

    def run(
        self,
        config: SensorConfig,
        sensor_models: dict,
        n_episodes: int = 15,
        rng: np.random.Generator | None = None,
    ) -> EvalMetrics:
        # Keep `run()` working even when the optimizer calls single evaluations.
        return self.run_batch(
            configs=[config],
            sensor_models=sensor_models,
            n_episodes=n_episodes,
            rng=rng,
        )[0]

    def run_batch(
        self,
        configs: list[SensorConfig],
        sensor_models: dict,
        n_episodes: int = 15,
        rng: np.random.Generator | None = None,
    ) -> list[EvalMetrics]:
        """
        Evaluate multiple configs using parallel Isaac Sim environments.

        Mechanism:
        - Process candidates in chunks of size `self.num_envs`.
        - For each chunk, reconfigure sensors for env slots [0..k-1] only.
        - Run rollouts for all envs simultaneously (Isaac-side vectorized stepping).
        - Collect metrics for the active env slots and append them in input order.

        Ordering:
        - Returned list matches `configs` order exactly, including the final partial chunk.

        Determinism:
        - Uses the provided `rng` as a seed source; each chunk gets a derived RNG so
          repeated runs with the same initial RNG state are reproducible.
        """
        if rng is None:
            rng = np.random.default_rng()

        if not configs:
            return []

        if self.env is None:
            raise NotImplementedError(
                "IsaacSimEvaluator requires an Isaac environment instance. "
                "Provide it via isaac_sim_cfg={'env': <your_env>, 'num_envs': N}."
            )

        out: list[EvalMetrics] = []

        # Derive per-chunk RNGs deterministically from the provided generator.
        # This avoids accidental dependence on internal Isaac stepping order.
        for _, chunk in _chunked(list(configs), self.num_envs):
            k = len(chunk)

            # (a) Reconfigure each parallel environment with its SensorConfig.
            for env_idx, cfg in enumerate(chunk):
                self.env.reconfigure_sensors(env_idx, cfg, sensor_models)

            # (b) Run rollouts for all environments simultaneously.
            # Isaac-side should do vectorized stepping (minimize Python loops).
            # We pass a derived RNG for chunk-level determinism.
            chunk_seed = int(rng.integers(0, np.iinfo(np.int32).max))
            chunk_rng = np.random.default_rng(chunk_seed)
            metrics_all = _call_run_rollouts(
                self.env,
                n_episodes=n_episodes,
                rng=chunk_rng,
                sensor_noise_std=self._sensor_noise_std,
            )

            if not isinstance(metrics_all, list):
                raise TypeError("env.run_rollouts(...) must return list[EvalMetrics]")
            if len(metrics_all) < k:
                raise ValueError(
                    "env.run_rollouts(...) returned fewer metrics than active envs "
                    f"(got {len(metrics_all)}, need >= {k})."
                )

            # (c) Collect EvalMetrics for each config (only the active env slots).
            out.extend(metrics_all[:k])

        return out


def _call_run_rollouts(env: object, n_episodes: int, rng: np.random.Generator, sensor_noise_std: float) -> list:
    """
    Call env.run_rollouts, passing sensor_noise_std when supported (e.g. Colab JSON bridge).
    """
    try:
        return env.run_rollouts(  # type: ignore[call-arg]
            n_episodes=n_episodes,
            rng=rng,
            sensor_noise_std=float(sensor_noise_std),
        )
    except TypeError:
        return env.run_rollouts(n_episodes=n_episodes, rng=rng)  # type: ignore[call-arg]


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