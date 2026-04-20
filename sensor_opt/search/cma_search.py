from __future__ import annotations

from sensor_opt.cma.outer_loop import run_cma_optimization
from sensor_opt.search.base import BaseSearch


class CMASearch(BaseSearch):
    """
    CMA wrapper that delegates to the existing outer-loop implementation.
    """

    def _wrap_evaluator(self):
        """
        Adapter kept as an explicit extension point for future scalar-only
        optimizers. Current CMA path uses structured evaluator objects directly
        through run_cma_optimization / run_outer_loop.
        """

        def fn(config):
            result = self.evaluator.evaluate(config)
            return self._scalarize(result)

        return fn

    def _scalarize(self, result):
        # maintain compatibility with existing loss behavior
        if hasattr(result, "loss"):
            loss = result.loss
            return loss.total if hasattr(loss, "total") else loss

        # fallback: weighted sum
        return (
            result.objectives["collision"]
            + result.objectives["blind_spot"]
            + result.objectives["cost"]
        )

    def run(self):
        logger = self.evaluator.get("logger")
        seed = int(self.evaluator.get("seed", 42))
        evaluator_fn = self.evaluator.get("evaluator_fn")
        evaluator_obj = self.evaluator.get("evaluator")
        base_evaluator = self.evaluator.get("base_evaluator")

        return run_cma_optimization(
            config=self.config,
            evaluator=evaluator_fn,
            logger=logger,
            seed=seed,
            evaluator_obj=evaluator_obj,
            base_evaluator=base_evaluator,
        )
