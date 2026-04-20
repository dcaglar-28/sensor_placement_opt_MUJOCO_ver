"""
sensor_opt/cma/outer_loop.py

CMA-ES outer loop. Wraps pycma to:
  1. Encode/decode sensor configs
  2. Evaluate each candidate via the inner loop
  3. Compute loss
  4. Log results
  5. Return the best config found
"""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Callable, List, Optional

import cma
import numpy as np

from sensor_opt.cma.pareto import ParetoPoint, pareto_front
from sensor_opt.encoding.config import (
    SensorConfig,
    decode,
    make_initial_vector,
)
from sensor_opt.evaluation.base import BaseEvaluator
from sensor_opt.evaluation.pipeline import Evaluator
from sensor_opt.evaluation.results import EvaluationResult
from sensor_opt.loss.loss import EvalMetrics, LossResult, compute_loss
from sensor_opt.logging.experiment_logger import ExperimentLogger


@dataclass
class OptimizationResult:
    best_config: SensorConfig
    best_loss: float
    best_loss_result: LossResult
    pareto_front: List[ParetoPoint]
    n_generations: int
    converged: bool
    stop_reason: str
    run_id: str


def run_outer_loop(
    cfg: dict,
    evaluator_fn: Optional[Callable[[SensorConfig, dict, int, float, Optional[np.random.Generator]], EvalMetrics]],
    logger: ExperimentLogger,
    seed: int = 42,
    evaluator: Optional[Evaluator] = None,
    base_evaluator: Optional[BaseEvaluator] = None,
) -> OptimizationResult:

    rng = np.random.default_rng(seed)

    sensor_budget  = cfg["sensor_budget"]
    mounting_slots = cfg["mounting_slots"]
    sensor_models  = cfg["sensor_models"]
    cma_cfg        = cfg["cma"]
    loss_cfg       = cfg["loss"]
    inner_cfg      = cfg["inner_loop"]

    x0  = make_initial_vector(sensor_budget, mounting_slots)
    dim = len(x0)
    print(f"[CMA-ES] Vector dimension: {dim} ({dim // 10} sensor slots × 10 params)")

    pop_size = cma_cfg.get("population_size", None)
    cma_options = {
        "seed":    seed,
        "tolx":    cma_cfg.get("tolx", 1e-4),
        "tolfun":  cma_cfg.get("tolfun", 1e-5),
        "maxiter": cma_cfg.get("max_generations", 100),
        "verbose": -9,
    }
    if pop_size is not None:
        cma_options["popsize"] = pop_size

    es = cma.CMAEvolutionStrategy(x0, float(cma_cfg.get("sigma0", 0.3)), cma_options)

    best_loss   = float("inf")
    best_config = None
    best_result = None
    generation  = 0
    global_configs: List[SensorConfig] = []
    global_objectives: List[dict] = []

    noise_std  = inner_cfg.get("dummy", {}).get("noise_std", 0.05)
    n_episodes = inner_cfg.get("n_episodes", 15)

    while not es.stop():
        generation += 1
        solutions  = es.ask()

        losses: List[float] = []
        loss_results: List[LossResult] = []
        eval_times: List[float] = []
        fidelities: List[str] = []

        for vec in solutions:
            config = decode(vec, mounting_slots, sensor_budget)

            try:
                eval_result = _evaluate_candidate(
                    config=config,
                    cfg=cfg,
                    sensor_models=sensor_models,
                    loss_cfg=loss_cfg,
                    n_episodes=n_episodes,
                    noise_std=noise_std,
                    rng=rng,
                    evaluator_fn=evaluator_fn,
                    evaluator=evaluator,
                    base_evaluator=base_evaluator,
                )
            except Exception as e:
                print(f"[CMA-ES] Evaluator error: {e}")
                traceback.print_exc()
                fallback_metrics = EvalMetrics(
                    collision_rate=1.0,
                    blind_spot_fraction=1.0,
                    mean_goal_success=0.0,
                    n_episodes=n_episodes,
                )
                fallback_lr = compute_loss(
                    metrics=fallback_metrics,
                    config=config,
                    sensor_models=sensor_models,
                    weights={
                        "alpha": loss_cfg["alpha"],
                        "beta":  loss_cfg["beta"],
                        "gamma": loss_cfg["gamma"],
                    },
                    max_cost_usd=loss_cfg.get("max_cost_usd", 10_000.0),
                    hardware_constraints=cfg.get("hardware", {}),
                )
                eval_result = EvaluationResult(
                    metrics=fallback_metrics,
                    loss=fallback_lr,
                    objectives=dict(fallback_lr.objectives or {}),
                    fidelity="fallback",
                    evaluation_time_sec=0.0,
                )

            lr = eval_result.loss
            losses.append(lr.total)
            loss_results.append(lr)
            eval_times.append(float(eval_result.evaluation_time_sec))
            fidelities.append(eval_result.fidelity)

            global_configs.append(config)
            global_objectives.append(dict(eval_result.objectives))

        es_losses = list(losses)
        if es_losses and float(np.max(es_losses) - np.min(es_losses)) == 0.0:
            # Avoid immediate tolfun stop on fully tied generations (common when all
            # decoded configs are disabled early in search). Keep jitter tiny so it
            # only breaks ties for CMA ranking without changing logged objectives.
            eps = 1e-9
            es_losses = [v + i * eps for i, v in enumerate(es_losses)]

        es.tell(solutions, es_losses)

        gen_best_idx = int(np.argmin(losses))
        gen_best_loss = losses[gen_best_idx]
        gen_best_lr   = loss_results[gen_best_idx]

        if gen_best_loss < best_loss:
            best_loss   = gen_best_loss
            best_config = decode(solutions[gen_best_idx], mounting_slots, sensor_budget)
            best_result = gen_best_lr

        log_every = cfg.get("logging", {}).get("log_every_n_generations", 1)
        if generation % log_every == 0:
            logger.log_generation(
                generation=generation,
                losses=losses,
                best_result=gen_best_lr,
                cma_sigma=float(es.sigma),
                mean_eval_time_sec=float(np.mean(eval_times) if eval_times else 0.0),
                dominant_fidelity=_dominant_label(fidelities),
            )
            _print_progress(generation, gen_best_loss, best_loss, es.sigma, gen_best_lr)

    stop_reason = str(es.stop())
    converged   = "tolfun" in stop_reason or "tolx" in stop_reason

    print(f"\n[CMA-ES] Stopped after {generation} generations.")
    print(f"[CMA-ES] Stop reason: {stop_reason}")
    print(f"[CMA-ES] Best loss: {best_loss:.6f}")
    if best_result:
        print(f"[CMA-ES] Best config: {best_result.config_summary}")
        print(f"[CMA-ES]   Collision: {best_result.collision_term:.4f} "
              f"| Blind: {best_result.blind_term:.4f} "
              f"| Cost: ${best_result.cost_usd:.0f}")

    final_pareto = pareto_front(global_configs, global_objectives)
    print(f"[CMA-ES] Pareto-front size: {len(final_pareto)}")

    return OptimizationResult(
        best_config=best_config,
        best_loss=best_loss,
        best_loss_result=best_result,
        pareto_front=final_pareto,
        n_generations=generation,
        converged=converged,
        stop_reason=stop_reason,
        run_id=logger.run_id,
    )


def run_cma_optimization(
    config: dict,
    evaluator,
    logger: ExperimentLogger,
    seed: int = 42,
    evaluator_obj: Optional[Evaluator] = None,
    base_evaluator: Optional[BaseEvaluator] = None,
) -> OptimizationResult:
    """
    Compatibility wrapper for pluggable search architecture.
    Reuses existing CMA outer-loop implementation unchanged.
    """
    return run_outer_loop(
        cfg=config,
        evaluator_fn=evaluator,
        logger=logger,
        seed=seed,
        evaluator=evaluator_obj,
        base_evaluator=base_evaluator,
    )


def _print_progress(gen, gen_best, all_best, sigma, lr):
    print(
        f"Gen {gen:4d} | gen_best={gen_best:.4f} | all_best={all_best:.4f} "
        f"| σ={sigma:.4f} | active={lr.n_active_sensors} "
        f"| ${lr.cost_usd:.0f} | {lr.config_summary}"
    )


def _evaluate_candidate(
    config: SensorConfig,
    cfg: dict,
    sensor_models: dict,
    loss_cfg: dict,
    n_episodes: int,
    noise_std: float,
    rng: np.random.Generator,
    evaluator_fn,
    evaluator: Optional[Evaluator],
    base_evaluator: Optional[BaseEvaluator],
) -> EvaluationResult:
    if evaluator is not None:
        return evaluator.evaluate(config=config, n_episodes=n_episodes, rng=rng, cfg=cfg)

    if base_evaluator is not None:
        metrics = base_evaluator.run(
            config=config,
            sensor_models=sensor_models,
            n_episodes=n_episodes,
            rng=rng,
        )
    else:
        if evaluator_fn is None:
            raise ValueError("No evaluator provided. Pass evaluator_fn, base_evaluator, or evaluator.")
        metrics = evaluator_fn(config, sensor_models, n_episodes, noise_std, rng)

    lr = compute_loss(
        metrics=metrics,
        config=config,
        sensor_models=sensor_models,
        weights={
            "alpha": loss_cfg["alpha"],
            "beta":  loss_cfg["beta"],
            "gamma": loss_cfg["gamma"],
        },
        max_cost_usd=loss_cfg.get("max_cost_usd", 10_000.0),
        hardware_constraints=cfg.get("hardware", {}),
    )
    return EvaluationResult(
        metrics=metrics,
        loss=lr,
        objectives=dict(lr.objectives or {}),
        fidelity="single",
        evaluation_time_sec=0.0,
    )


def _dominant_label(labels: List[str]) -> str:
    if not labels:
        return "n/a"
    uniq, counts = np.unique(np.array(labels), return_counts=True)
    return str(uniq[int(np.argmax(counts))])