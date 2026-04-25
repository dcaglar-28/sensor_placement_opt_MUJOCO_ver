from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from sensor_opt.cma.outer_loop import OptimizationResult
from sensor_opt.cma.pareto import pareto_front
from sensor_opt.design.config import DesignConfig, build_design_config
from sensor_opt.evaluation.results import EvaluationResult
from sensor_opt.loss.loss import EvalMetrics, compute_loss, loss_weight_dict
from sensor_opt.search.base import BaseSearch
from sensor_opt.search.encoding import make_config_encoder

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
except Exception:  # pragma: no cover
    GaussianProcessRegressor = None
    Matern = None
    WhiteKernel = None


@dataclass
class _EvalRecord:
    design: DesignConfig
    result: EvaluationResult
    score: float


class BayesianSearch(BaseSearch):
    def run(self):
        cfg = self.config
        bcfg = cfg.get("bayesian", {})
        init_samples = int(bcfg.get("init_samples", 20))
        iterations = int(bcfg.get("iterations", 50))
        candidates_per_iter = int(bcfg.get("acq_candidates", 64))
        seed = int(self.evaluator.get("seed", 42))
        rng = np.random.default_rng(seed)
        logger = self.evaluator.get("logger")

        encoder = make_config_encoder(cfg)
        samples: List[_EvalRecord] = [self._random_record(rng, encoder) for _ in range(init_samples)]
        model = self._make_model()

        for step in range(1, iterations + 1):
            X = np.stack([encoder.encode(s.design.sensors) for s in samples], axis=0)
            y = np.array([s.score for s in samples], dtype=float)
            model.fit(X, y)

            candidates = [self._random_config(rng, encoder) for _ in range(candidates_per_iter)]
            cand_X = np.stack([encoder.encode(c) for c in candidates], axis=0)
            acq = self._ucb(model, cand_X, kappa=float(bcfg.get("kappa", 1.25)))
            next_cfg = candidates[int(np.argmin(acq))]
            rec = self._evaluate(next_cfg, rng)
            samples.append(rec)

            if logger is not None:
                losses = [s.result.loss.total for s in samples[-min(20, len(samples)):]]
                logger.log_generation(
                    generation=step,
                    losses=losses,
                    best_result=min(samples, key=lambda s: s.result.loss.total).result.loss,
                    cma_sigma=0.0,
                    mean_eval_time_sec=float(np.mean([s.result.evaluation_time_sec for s in samples[-min(20, len(samples)):]])),
                    dominant_fidelity=self._dominant_fidelity(samples[-min(20, len(samples)):]),
                )

        best = min(samples, key=lambda s: s.result.loss.total)
        pareto_pts = pareto_front([s.design.sensors for s in samples], [s.result.objectives for s in samples])
        return OptimizationResult(
            best_config=best.design.sensors,
            best_loss=best.result.loss.total,
            best_loss_result=best.result.loss,
            pareto_front=pareto_pts,
            n_generations=iterations,
            converged=False,
            stop_reason="bayesian_iterations",
            run_id=logger.run_id if logger is not None else "bayes_run",
        )

    def _make_model(self):
        if GaussianProcessRegressor is None:
            raise ImportError("scikit-learn is required for BayesianSearch.")
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)

    def _ucb(self, model, X: np.ndarray, kappa: float) -> np.ndarray:
        mu, sigma = model.predict(X, return_std=True)
        return mu - kappa * sigma

    def _random_config(self, rng: np.random.Generator, encoder: ConfigEncoder):
        dim = sum(v.get("max_count", 0) for v in self.config["sensor_budget"].values()) * 10
        vec = rng.uniform(-1.0, 3.5, size=dim)
        return encoder.decode(vec)

    def _random_record(self, rng: np.random.Generator, encoder: ConfigEncoder) -> _EvalRecord:
        return self._evaluate(self._random_config(rng, encoder), rng)

    def _evaluate(self, sensor_cfg, rng: np.random.Generator) -> _EvalRecord:
        design = build_design_config(sensor_cfg, self.config)
        result = self._evaluate_design(design, rng)
        score = self._scalarize(result)
        return _EvalRecord(design=design, result=result, score=score)

    def _evaluate_design(self, design: DesignConfig, rng: np.random.Generator) -> EvaluationResult:
        if self.evaluator.get("evaluator") is not None:
            return self.evaluator["evaluator"].evaluate(
                config=design.sensors,
                n_episodes=self.config["inner_loop"].get("n_episodes", 15),
                rng=rng,
                cfg=self.config,
            )
        sensor_models = self.config["sensor_models"]
        n_episodes = self.config["inner_loop"].get("n_episodes", 15)
        noise_std = self.config["inner_loop"].get("dummy", {}).get("noise_std", 0.05)
        base_eval = self.evaluator.get("base_evaluator")
        eval_fn = self.evaluator.get("evaluator_fn")
        if base_eval is not None:
            metrics = base_eval.run(design.sensors, sensor_models, n_episodes=n_episodes, rng=rng)
        else:
            metrics = eval_fn(design.sensors, sensor_models, n_episodes, noise_std, rng)
        return self._build_eval_result(metrics, design.sensors)

    def _build_eval_result(self, metrics: EvalMetrics, sensor_cfg) -> EvaluationResult:
        loss_cfg = self.config["loss"]
        lr = compute_loss(
            metrics=metrics,
            config=sensor_cfg,
            sensor_models=self.config["sensor_models"],
            weights=loss_weight_dict(loss_cfg),
            max_cost_usd=loss_cfg.get("max_cost_usd", 10_000.0),
            hardware_constraints=self.config.get("hardware", {}),
            loss_mode=str(loss_cfg.get("mode", "default")),
        )
        return EvaluationResult(
            metrics=metrics,
            loss=lr,
            objectives=dict(lr.objectives or {}),
            fidelity="single",
            evaluation_time_sec=0.0,
        )

    def _scalarize(self, result: EvaluationResult) -> float:
        if hasattr(result, "loss") and hasattr(result.loss, "total"):
            return float(result.loss.total)
        obj = result.objectives
        return float(obj["collision"] + obj["blind_spot"] + obj["cost"] + obj.get("hardware", 0.0))

    @staticmethod
    def _dominant_fidelity(samples: List[_EvalRecord]) -> str:
        labels = [s.result.fidelity for s in samples]
        uniq, counts = np.unique(np.array(labels), return_counts=True)
        return str(uniq[int(np.argmax(counts))]) if len(uniq) else "n/a"
