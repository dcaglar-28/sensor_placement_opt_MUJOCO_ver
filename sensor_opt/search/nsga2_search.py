from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from sensor_opt.cma.outer_loop import OptimizationResult
from sensor_opt.cma.pareto import ParetoPoint, pareto_front
from sensor_opt.design.config import DesignConfig, build_design_config
from sensor_opt.encoding.config import SensorConfig, config_vector_size, decode
from sensor_opt.evaluation.results import EvaluationResult
from sensor_opt.loss.loss import EvalMetrics, compute_loss, loss_weight_dict
from sensor_opt.search.base import BaseSearch


@dataclass
class _Individual:
    design: DesignConfig
    result: EvaluationResult
    rank: int = 0
    crowding: float = 0.0


class NSGA2Search(BaseSearch):
    def run(self):
        cfg = self.config
        search_cfg = cfg.get("nsga2", {})
        pop_size = int(search_cfg.get("population_size", 32))
        generations = int(search_cfg.get("generations", 20))
        seed = int(self.evaluator.get("seed", 42))
        rng = np.random.default_rng(seed)
        logger = self.evaluator.get("logger")

        population = self._initial_population(pop_size, rng)
        best_design = population[0]
        best_loss = float("inf")
        best_lr = None
        all_designs: List[DesignConfig] = []
        all_objectives: List[Dict[str, float]] = []

        for gen in range(1, generations + 1):
            evaluated = [self._evaluate_design(d, rng) for d in population]
            self._assign_ranks_and_crowding(evaluated)
            losses = [ind.result.loss.total for ind in evaluated]
            gen_best = min(evaluated, key=lambda i: i.result.loss.total)
            if gen_best.result.loss.total < best_loss:
                best_loss = gen_best.result.loss.total
                best_design = gen_best.design
                best_lr = gen_best.result.loss

            all_designs.extend([ind.design for ind in evaluated])
            all_objectives.extend([ind.result.objectives for ind in evaluated])

            if logger is not None and best_lr is not None:
                logger.log_generation(
                    generation=gen,
                    losses=losses,
                    best_result=gen_best.result.loss,
                    cma_sigma=0.0,
                    mean_eval_time_sec=float(np.mean([i.result.evaluation_time_sec for i in evaluated])),
                    dominant_fidelity=self._dominant_fidelity(evaluated),
                )

            parents = self._tournament_select(evaluated, pop_size, rng)
            offspring = self._make_offspring(parents, pop_size, rng)
            combined = evaluated + [self._evaluate_design(d, rng) for d in offspring]
            population = self._truncate(combined, pop_size)

        pareto_pts = pareto_front([d.sensors for d in all_designs], all_objectives)
        return OptimizationResult(
            best_config=best_design.sensors,
            best_loss=best_loss,
            best_loss_result=best_lr,
            pareto_front=pareto_pts,
            n_generations=generations,
            converged=False,
            stop_reason="max_generations",
            run_id=logger.run_id if logger is not None else "nsga2_run",
        )

    def _initial_population(self, pop_size: int, rng: np.random.Generator) -> List[DesignConfig]:
        budget = self.config["sensor_budget"]
        slots = self.config["mounting_slots"]
        dim = config_vector_size(budget)
        pop: List[DesignConfig] = []
        for _ in range(pop_size):
            vec = rng.uniform(-1.0, 3.5, size=dim)
            sensor_cfg = decode(vec, slots, budget)
            pop.append(build_design_config(sensor_cfg, self.config))
        return pop

    def _evaluate_design(self, design: DesignConfig, rng: np.random.Generator) -> _Individual:
        if self.evaluator.get("evaluator") is not None:
            res = self.evaluator["evaluator"].evaluate(
                config=design.sensors,
                n_episodes=self.config["inner_loop"].get("n_episodes", 15),
                rng=rng,
                cfg=self.config,
            )
            return _Individual(design=design, result=res)

        sensor_models = self.config["sensor_models"]
        n_episodes = self.config["inner_loop"].get("n_episodes", 15)
        noise_std = self.config["inner_loop"].get("dummy", {}).get("noise_std", 0.05)
        base_eval = self.evaluator.get("base_evaluator")
        eval_fn = self.evaluator.get("evaluator_fn")
        if base_eval is not None:
            metrics = base_eval.run(design.sensors, sensor_models, n_episodes=n_episodes, rng=rng)
        else:
            metrics = eval_fn(design.sensors, sensor_models, n_episodes, noise_std, rng)
        result = self._build_eval_result(metrics, design.sensors)
        return _Individual(design=design, result=result)

    def _build_eval_result(self, metrics: EvalMetrics, sensor_cfg: SensorConfig) -> EvaluationResult:
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

    def _assign_ranks_and_crowding(self, population: List[_Individual]) -> None:
        obj = [p.result.objectives for p in population]
        fronts = self.fast_non_dominated_sort(population, obj)
        for rank, front in enumerate(fronts):
            dists = self.crowding_distance(front, obj)
            for idx, dist in zip(front, dists):
                population[idx].rank = rank
                population[idx].crowding = dist

    def fast_non_dominated_sort(self, population: List[_Individual], objectives: List[Dict[str, float]]) -> List[List[int]]:
        n = len(population)
        dominates_list = [[] for _ in range(n)]
        dominated_count = [0] * n
        fronts: List[List[int]] = [[]]
        keys = ["collision", "blind_spot", "cost", "hardware"]

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(objectives[i], objectives[j], keys):
                    dominates_list[i].append(j)
                elif self._dominates(objectives[j], objectives[i], keys):
                    dominated_count[i] += 1
            if dominated_count[i] == 0:
                fronts[0].append(i)

        f = 0
        while f < len(fronts) and fronts[f]:
            next_front = []
            for i in fronts[f]:
                for j in dominates_list[i]:
                    dominated_count[j] -= 1
                    if dominated_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            f += 1
        return fronts

    def crowding_distance(self, front: List[int], objectives: List[Dict[str, float]]) -> List[float]:
        if not front:
            return []
        if len(front) <= 2:
            return [float("inf")] * len(front)
        keys = ["collision", "blind_spot", "cost"]
        dist = np.zeros(len(front), dtype=float)
        front_arr = np.array(front)
        for key in keys:
            vals = np.array([objectives[i].get(key, 0.0) for i in front], dtype=float)
            order = np.argsort(vals)
            dist[order[0]] = float("inf")
            dist[order[-1]] = float("inf")
            vmin, vmax = vals[order[0]], vals[order[-1]]
            denom = max(vmax - vmin, 1e-9)
            for k in range(1, len(front) - 1):
                if np.isinf(dist[order[k]]):
                    continue
                prev_v = vals[order[k - 1]]
                next_v = vals[order[k + 1]]
                dist[order[k]] += (next_v - prev_v) / denom
        return dist.tolist()

    def _tournament_select(self, population: List[_Individual], count: int, rng: np.random.Generator) -> List[DesignConfig]:
        selected: List[DesignConfig] = []
        for _ in range(count):
            i, j = rng.integers(0, len(population), size=2)
            a, b = population[i], population[j]
            winner = a if (a.rank < b.rank or (a.rank == b.rank and a.crowding >= b.crowding)) else b
            selected.append(winner.design)
        return selected

    def _make_offspring(self, parents: List[DesignConfig], count: int, rng: np.random.Generator) -> List[DesignConfig]:
        out: List[DesignConfig] = []
        for _ in range(count):
            p1 = parents[int(rng.integers(0, len(parents)))]
            p2 = parents[int(rng.integers(0, len(parents)))]
            child_sensor = self._crossover(p1.sensors, p2.sensors, rng)
            child_sensor = self._mutate(child_sensor, rng)
            out.append(build_design_config(child_sensor, self.config))
        return out

    def _truncate(self, combined: List[_Individual], pop_size: int) -> List[DesignConfig]:
        self._assign_ranks_and_crowding(combined)
        combined_sorted = sorted(combined, key=lambda i: (i.rank, -i.crowding))
        return [ind.design for ind in combined_sorted[:pop_size]]

    def _crossover(self, a: SensorConfig, b: SensorConfig, rng: np.random.Generator) -> SensorConfig:
        sensors = []
        n = max(len(a.sensors), len(b.sensors))
        for idx in range(n):
            pick_a = bool(rng.random() < 0.5)
            src = a.sensors if pick_a else b.sensors
            if idx < len(src):
                s = src[idx]
            else:
                other = b.sensors if pick_a else a.sensors
                s = other[min(idx, len(other) - 1)]
            sensors.append(type(s)(**vars(s)))
        return SensorConfig(sensors=sensors)

    def _mutate(self, cfg: SensorConfig, rng: np.random.Generator) -> SensorConfig:
        budget = self.config["sensor_budget"]
        allowed_types = list(budget.keys()) + ["disabled"]
        for sensor in cfg.sensors:
            if rng.random() < 0.35:
                sensor.x_offset = float(np.clip(sensor.x_offset + rng.normal(0.0, 0.08), -0.5, 0.5))
                sensor.y_offset = float(np.clip(sensor.y_offset + rng.normal(0.0, 0.08), -0.5, 0.5))
                sensor.z_offset = float(np.clip(sensor.z_offset + rng.normal(0.0, 0.05), 0.0, 0.5))
            if rng.random() < 0.20:
                sensor.sensor_type = str(allowed_types[int(rng.integers(0, len(allowed_types)))])
            if rng.random() < 0.15:
                sensor.range_fraction = float(np.clip(sensor.range_fraction + rng.normal(0.0, 0.1), 0.1, 1.0))
            if rng.random() < 0.15:
                sensor.hfov_fraction = float(np.clip(sensor.hfov_fraction + rng.normal(0.0, 0.1), 0.2, 1.0))
        # Enforce simple add/remove pressure via random disable/enable toggles.
        if cfg.sensors and rng.random() < 0.15:
            idx = int(rng.integers(0, len(cfg.sensors)))
            cfg.sensors[idx].sensor_type = "disabled"
        if cfg.sensors and rng.random() < 0.15:
            idx = int(rng.integers(0, len(cfg.sensors)))
            if cfg.sensors[idx].sensor_type == "disabled":
                cfg.sensors[idx].sensor_type = str(allowed_types[int(rng.integers(0, len(allowed_types) - 1))])
        return cfg

    @staticmethod
    def _dominates(a: Dict[str, float], b: Dict[str, float], keys: List[str]) -> bool:
        no_worse = all(a.get(k, 0.0) <= b.get(k, 0.0) for k in keys)
        strictly = any(a.get(k, 0.0) < b.get(k, 0.0) for k in keys)
        return no_worse and strictly

    @staticmethod
    def _dominant_fidelity(evaluated: List[_Individual]) -> str:
        labels = [i.result.fidelity for i in evaluated]
        uniq, counts = np.unique(np.array(labels), return_counts=True)
        return str(uniq[int(np.argmax(counts))]) if len(uniq) else "n/a"
