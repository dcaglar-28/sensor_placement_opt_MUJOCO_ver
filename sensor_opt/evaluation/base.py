"""
Common evaluator interface for swappable fidelity backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Optional

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.loss.loss import EvalMetrics


class BaseEvaluator(ABC):
    """Abstract evaluator contract used by outer-loop search."""

    @abstractmethod
    def run(
        self,
        config: SensorConfig,
        sensor_models: dict,
        n_episodes: int = 15,
        rng: Optional[np.random.Generator] = None,
    ) -> EvalMetrics:
        raise NotImplementedError

    def run_batch(
        self,
        configs: Iterable[SensorConfig],
        sensor_models: dict,
        n_episodes: int = 15,
        rng: Optional[np.random.Generator] = None,
    ) -> list[EvalMetrics]:
        """
        Batched evaluation hook.

        Default implementation falls back to a Python loop to preserve backward
        compatibility. Backends (including Isaac Sim) can override this to do
        true batching / vectorized simulation.
        """
        return [
            self.run(config=c, sensor_models=sensor_models, n_episodes=n_episodes, rng=rng)
            for c in list(configs)
        ]
