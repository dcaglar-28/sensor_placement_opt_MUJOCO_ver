"""
sensor_opt/logging/experiment_logger.py

CSV + MLflow experiment tracking.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np

from sensor_opt.encoding.serialize_config import sensor_config_to_dict
from sensor_opt.loss.loss import LossResult, _compute_effective_cost

if TYPE_CHECKING:
    from sensor_opt.cma.pareto import ParetoPoint
    from sensor_opt.encoding.config import SensorConfig


@dataclass
class GenerationRecord:
    run_id: str
    experiment_name: str
    generation: int
    elapsed_sec: float
    best_loss: float
    mean_loss: float
    std_loss: float
    best_collision_term: float
    best_blind_term: float
    best_cost_term: float
    best_cost_usd: float
    best_n_active: int
    best_config_summary: str
    population_size: int
    cma_sigma: float
    mean_eval_time_sec: float = 0.0
    dominant_fidelity: str = "single"


class ExperimentLogger:
    CSV_FIELDNAMES = [
        "run_id", "experiment_name", "generation", "elapsed_sec",
        "best_loss", "mean_loss", "std_loss",
        "best_collision_term", "best_blind_term", "best_cost_term",
        "best_cost_usd", "best_n_active", "best_config_summary",
        "population_size", "cma_sigma", "mean_eval_time_sec", "dominant_fidelity",
    ]

    def __init__(
        self,
        experiment_name: str,
        results_dir: str = "results",
        use_mlflow: bool = True,
        mlflow_tracking_uri: str = "mlruns",
        run_config: Optional[dict] = None,
    ):
        self.experiment_name = experiment_name
        self.results_dir = Path(results_dir)
        self.use_mlflow = use_mlflow
        self.run_config = run_config or {}
        self._start_time = time.time()

        ts = time.strftime("%Y%m%d_%H%M%S")
        self.run_id  = f"{experiment_name}_{ts}"
        self.run_dir = self.results_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path   = self.run_dir / "generations.csv"
        self._csv_file  = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self.CSV_FIELDNAMES)
        self._csv_writer.writeheader()

        with open(self.run_dir / "config.json", "w") as f:
            json.dump(run_config, f, indent=2)

        self._mlflow_run = None
        if use_mlflow:
            self._setup_mlflow(mlflow_tracking_uri)

        self.records: List[GenerationRecord] = []

    def log_generation(
        self,
        generation: int,
        losses: List[float],
        best_result: LossResult,
        cma_sigma: float,
        mean_eval_time_sec: float = 0.0,
        dominant_fidelity: str = "single",
    ) -> None:
        arr = np.array(losses)
        record = GenerationRecord(
            run_id=self.run_id,
            experiment_name=self.experiment_name,
            generation=generation,
            elapsed_sec=round(time.time() - self._start_time, 2),
            best_loss=round(float(arr.min()), 6),
            mean_loss=round(float(arr.mean()), 6),
            std_loss=round(float(arr.std()), 6),
            best_collision_term=round(best_result.collision_term, 6),
            best_blind_term=round(best_result.blind_term, 6),
            best_cost_term=round(best_result.cost_term, 6),
            best_cost_usd=round(best_result.cost_usd, 2),
            best_n_active=best_result.n_active_sensors,
            best_config_summary=best_result.config_summary,
            population_size=len(losses),
            cma_sigma=round(float(cma_sigma), 6),
            mean_eval_time_sec=round(float(mean_eval_time_sec), 6),
            dominant_fidelity=dominant_fidelity,
        )
        self.records.append(record)
        self._csv_writer.writerow(asdict(record))
        self._csv_file.flush()

        if self._mlflow_run is not None:
            self._log_mlflow(record)

    def log_final(self, best_result: LossResult, best_config_dict: dict) -> None:
        summary = {
            "run_id": self.run_id,
            "total_generations": len(self.records),
            "best_loss": best_result.total,
            "best_config": best_config_dict,
            "best_loss_breakdown": {
                "collision": best_result.collision_term,
                "blind_spot": best_result.blind_term,
                "cost": best_result.cost_term,
                "hardware_penalty": best_result.hardware_penalty_term,
                "cost_usd": best_result.cost_usd,
            },
            "best_objectives": best_result.objectives or {},
        }
        with open(self.run_dir / "final_result.json", "w") as f:
            json.dump(summary, f, indent=2)

        if self._mlflow_run is not None:
            try:
                import mlflow
                mlflow.log_dict(summary, "final_result.json")
                mlflow.end_run()
            except Exception:
                pass

    def log_paper_artifacts(
        self,
        *,
        global_configs: List["SensorConfig"],
        global_objectives: List[dict],
        eval_generations: List[int],
        pareto_front: List["ParetoPoint"],
        cfg: dict,
    ) -> None:
        """
        Write JSON used by `sensor_opt.plotting.paper_figures` (Pareto, correlation, violins).

        - evaluated_pool.json: every CMA candidate with generation index + objectives + pose params
        - pareto_front.json: non-dominated set with n_active + cost_usd
        - optimization_meta.json: pop size, total evaluations (for sample-efficiency figures)
        """
        sensor_models = cfg.get("sensor_models", {}) or {}
        cma_cfg = cfg.get("cma", {}) or {}
        pop_size = int(cma_cfg.get("population_size", 0) or 0)
        n_gen = 0
        if self.records:
            n_gen = max(r.generation for r in self.records)

        pool: List[dict] = []
        n = min(
            len(eval_generations),
            len(global_configs),
            len(global_objectives),
        )
        for i in range(n):
            gen, conf, obj = int(eval_generations[i]), global_configs[i], global_objectives[i]
            cost = float(_compute_effective_cost(conf, sensor_models))
            row = {
                "generation": gen,
                "objectives": dict(obj),
                "n_active_sensors": len(conf.active_sensors()),
                "cost_usd": cost,
                "config": sensor_config_to_dict(conf),
            }
            pool.append(row)

        with open(self.run_dir / "evaluated_pool.json", "w") as f:
            json.dump(pool, f, indent=2)

        pf_out: List[dict] = []
        for p in pareto_front:
            c = p.config
            cost = float(_compute_effective_cost(c, sensor_models))
            pf_out.append(
                {
                    "index": int(p.index),
                    "objectives": dict(p.objectives or {}),
                    "n_active_sensors": len(c.active_sensors()),
                    "cost_usd": cost,
                    "config": sensor_config_to_dict(c),
                }
            )
        with open(self.run_dir / "pareto_front.json", "w") as f:
            json.dump(pf_out, f, indent=2)

        meta = {
            "run_id": self.run_id,
            "population_size": pop_size,
            "generations": n_gen,
            "total_function_evals": len(global_configs),
            "pareto_size": len(pareto_front),
        }
        with open(self.run_dir / "optimization_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    def close(self) -> None:
        self._csv_file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _setup_mlflow(self, tracking_uri: str) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.experiment_name)
            self._mlflow_run = mlflow.start_run(run_name=self.run_id)
            flat = _flatten_dict(self.run_config, max_depth=2)
            for k, v in flat.items():
                try:
                    mlflow.log_param(k, v)
                except Exception:
                    pass
        except ImportError:
            print("[Logger] MLflow not installed — CSV-only logging.")
            self._mlflow_run = None
        except Exception as e:
            print(f"[Logger] MLflow setup failed ({e}) — CSV-only logging.")
            self._mlflow_run = None

    def _log_mlflow(self, record: GenerationRecord) -> None:
        try:
            import mlflow
            mlflow.log_metrics(
                {
                    "best_loss":     record.best_loss,
                    "mean_loss":     record.mean_loss,
                    "std_loss":      record.std_loss,
                    "cma_sigma":     record.cma_sigma,
                    "best_cost_usd": record.best_cost_usd,
                    "best_n_active": float(record.best_n_active),
                },
                step=record.generation,
            )
        except Exception:
            pass


def _flatten_dict(d: dict, prefix: str = "", max_depth: int = 2, depth: int = 0) -> dict:
    items: Dict[str, Any] = {}
    for k, v in d.items():
        new_key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict) and depth < max_depth:
            items.update(_flatten_dict(v, new_key, max_depth, depth + 1))
        else:
            items[new_key] = v
    return items