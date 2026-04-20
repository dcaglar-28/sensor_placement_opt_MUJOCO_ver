"""
Common evaluator interface for swappable fidelity backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

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
