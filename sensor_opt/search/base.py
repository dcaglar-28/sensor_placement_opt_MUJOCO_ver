from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSearch(ABC):
    """
    Base search abstraction for outer-loop optimization.

    Future implementations:
    - NSGA-II (multi-objective evolutionary)
    - Bayesian optimization (with surrogate)
    - random search (baseline)
    """

    def __init__(self, config, evaluator):
        self.config = config
        self.evaluator = evaluator

    @abstractmethod
    def run(self):
        """
        Returns:
            OptimizationResult (must match existing structure)
        """
        pass
