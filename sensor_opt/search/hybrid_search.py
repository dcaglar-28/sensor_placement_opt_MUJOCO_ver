from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from sensor_opt.cma.outer_loop import OptimizationResult, run_cma_optimization
from sensor_opt.cma.pareto import pareto_front
from sensor_opt.design.config import DesignConfig, build_design_config
from sensor_opt.evaluation.results import EvaluationResult
from sensor_opt.loss.loss import EvalMetrics, compute_loss, loss_weight_dict
from sensor_opt.search.base import BaseSearch
from sensor_opt.search.encoding import ConfigEncoder

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel
except Exception:  # pragma: no cover
    GaussianProcessRegressor = None
    Matern = None
    WhiteKernel = None


@dataclass
class _Record:
    design: DesignConfig
    result: EvaluationResult
    score: float


class HybridSearch(BaseSearch):
    def run(self):
        cfg = self.config
        hcfg = cfg.get("hybrid", {})
        warmup_iters = int(hcfg.get("warmup_iters", 10))
        surrogate_top_k = int(hcfg.get("surrogate_top_k", 5))
        seed = int(self.evaluator.get("seed", 42))
        rng = np.random.default_rng(seed)
        logger = self.evaluator.get("logger")

        warm_cfg = deepcopy(cfg)
        warm_cfg.setdefault("cma", {})
        warm_cfg["cma"]["max_generations"] = warmup_iters
        warmup = run_cma_optimization(
            config=warm_cfg,
            evaluator=self.evaluator.get("evaluator_fn"),
            logger=logger,
            seed=seed,
            evaluator_obj=self.evaluator.get("evaluator"),
            base_evaluator=self.evaluator.get("base_evaluator"),
        )

        records: List[_Record] = []
        for p in warmup.pareto_front:
            design = build_design_config(p.config, cfg)
            result = self._evaluate_design(design, rng)
            records.append(_Record(design=design, result=result, score=self._scalarize(result)))

        if not records:
            design = build_design_config(warmup.best_config, cfg)
            result = self._evaluate_design(design, rng)
            records.append(_Record(design=design, result=result, score=self._scalarize(result)))

        model = self._make_model()
        encoder = ConfigEncoder(cfg["mounting_slots"], cfg["sensor_budget"])

        X = np.stack([encoder.encode(r.design.sensors) for r in records], axis=0)
        y = np.array([r.score for r in records], dtype=float)
        model.fit(X, y)

        best_current = min(records, key=lambda r: r.score).design.sensors
        proposals = self._propose_candidates(best_current, encoder, rng, n=max(32, surrogate_top_k * 8))
        cand_X = np.stack([encoder.encode(c) for c in proposals], axis=0)
        mu, sigma = model.predict(cand_X, return_std=True)
        acq = mu - float(hcfg.get("kappa", 1.2)) * sigma
        top_idx = np.argsort(acq)[:surrogate_top_k]

        for idx in top_idx:
            design = build_design_config(proposals[int(idx)], cfg)
            result = self._evaluate_design(design, rng)
            records.append(_Record(design=design, result=result, score=self._scalarize(result)))

        best = min(records, key=lambda r: r.result.loss.total)
        pareto_pts = pareto_front([r.design.sensors for r in records], [r.result.objectives for r in records])
        return OptimizationResult(
            best_config=best.design.sensors,
            best_loss=best.result.loss.total,
            best_loss_result=best.result.loss,
            pareto_front=pareto_pts,
            n_generations=warmup.n_generations + 1,
            converged=warmup.converged,
            stop_reason=f"hybrid_after_{warmup.stop_reason}",
            run_id=warmup.run_id,
        )

    def _make_model(self):
        if GaussianProcessRegressor is None:
            raise ImportError("scikit-learn is required for HybridSearch.")
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-5)
        return GaussianProcessRegressor(kernel=kernel, normalize_y=True, random_state=0)

    def _propose_candidates(self, best_cfg, encoder: ConfigEncoder, rng: np.random.Generator, n: int):
        base = encoder.encode(best_cfg)
        out = []
        for _ in range(n):
            vec = base + rng.normal(0.0, 0.15, size=base.shape[0])
            out.append(encoder.decode(vec))
        return out

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

    @staticmethod
    def _scalarize(result: EvaluationResult) -> float:
        return float(result.loss.total)
