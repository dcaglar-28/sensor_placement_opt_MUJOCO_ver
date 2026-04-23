"""
sensor_opt/run_experiment.py

Main entry point.

Usage:
    python -m sensor_opt.run_experiment --config configs/default.yaml --dummy
    python -m sensor_opt.run_experiment --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import sys

import yaml

from sensor_opt.evaluation.pipeline import Evaluator
from sensor_opt.inner_loop.isaac_evaluator import IsaacSimEvaluator
from sensor_opt.inner_loop.mock_isaac_evaluator import MockIsaacEvaluator
from sensor_opt.logging.experiment_logger import ExperimentLogger
from sensor_opt.loss.loss import loss_weight_dict
from sensor_opt.search.factory import create_search


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Sensor Placement Optimizer")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--dummy", action="store_true",
                        help="Alias for mock Isaac evaluator (no Isaac Sim required)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed override")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("search", {"type": "cma"})

    if args.dummy:
        cfg["inner_loop"]["mode"] = "mock_isaac"

    mode = cfg["inner_loop"]["mode"]
    seed = args.seed if args.seed is not None else cfg["experiment"].get("seed", 42)

    print(f"[Experiment] name    : {cfg['experiment']['name']}")
    print(f"[Experiment] mode    : {mode}")
    print(f"[Experiment] seed    : {seed}")
    print(f"[Experiment] config  : {args.config}")

    evaluator_fn = None
    evaluator_obj = None
    base_evaluator = None

    if mode == "mock_isaac" or mode == "dummy":
        base_evaluator = MockIsaacEvaluator(
            latency_sec=float(cfg["inner_loop"].get("mock_isaac", {}).get("latency_sec", 0.15)),
            stochastic_std=float(cfg["inner_loop"].get("mock_isaac", {}).get("stochastic_std", 0.03)),
            baseline_noise_std=float(cfg["inner_loop"].get("mock_isaac", {}).get("baseline_noise_std", 0.01)),
        )
        evaluator_fn = None
        print("[Experiment] Using MOCK Isaac evaluator")
    elif mode == "isaac_sim":
        base_evaluator = IsaacSimEvaluator(isaac_sim_cfg=cfg["inner_loop"].get("isaac_sim", {}))
        evaluator_fn = None
        print("[Experiment] Using Isaac Sim evaluator")
    else:
        print(f"[Experiment] Unknown mode '{mode}'. Use 'mock_isaac' or 'isaac_sim'.")
        sys.exit(1)

    mf_cfg = cfg.get("multi_fidelity", {})
    if mf_cfg.get("enabled", False):
        evaluator_obj = Evaluator(
            # With dummy evaluator removed, we use MockIsaac with zero latency as
            # the "fast" and "mid" stages (lower stochasticity).
            fast_eval=MockIsaacEvaluator(
                latency_sec=0.0,
                stochastic_std=float(mf_cfg.get("fast_stochastic_std", 0.02)),
                baseline_noise_std=float(mf_cfg.get("fast_baseline_noise_std", 0.08)),
            ),
            mid_eval=MockIsaacEvaluator(
                latency_sec=0.0,
                stochastic_std=float(mf_cfg.get("mid_stochastic_std", 0.015)),
                baseline_noise_std=float(mf_cfg.get("mid_baseline_noise_std", 0.04)),
            ),
            slow_eval=MockIsaacEvaluator(
                latency_sec=float(mf_cfg.get("slow_latency_sec", 0.15)),
                stochastic_std=float(mf_cfg.get("slow_stochastic_std", 0.03)),
                baseline_noise_std=float(mf_cfg.get("slow_baseline_noise_std", 0.01)),
            ),
            weights=loss_weight_dict(cfg["loss"]),
            sensor_models=cfg["sensor_models"],
            loss_mode=str(cfg["loss"].get("mode", "default")),
            max_cost_usd=float(cfg["loss"].get("max_cost_usd", 10_000.0)),
            fast_collision_threshold=float(mf_cfg.get("fast_collision_threshold", 0.95)),
            promising_collision_threshold=float(mf_cfg.get("promising_collision_threshold", 0.35)),
        )
        print("[Experiment] Multi-fidelity pipeline enabled (fast -> mid -> slow)")

    log_cfg = cfg.get("logging", {})
    use_mlflow = log_cfg.get("mlflow", True) and not args.no_mlflow

    with ExperimentLogger(
        experiment_name=cfg["experiment"]["name"],
        results_dir=log_cfg.get("results_dir", "results"),
        use_mlflow=use_mlflow,
        mlflow_tracking_uri=log_cfg.get("mlflow_tracking_uri", "mlruns"),
        run_config=cfg,
    ) as logger:
        search = create_search(
            search_type=cfg["search"]["type"],
            config=cfg,
            evaluator={
                "evaluator_fn": evaluator_fn,
                "evaluator": evaluator_obj,
                "base_evaluator": base_evaluator,
                "logger": logger,
                "seed": seed,
            },
        )
        result = search.run()

        best_config_dict = {
            "sensors": [
                {
                    "type":           s.sensor_type,
                    "slot":           s.slot,
                    "x_offset":       round(s.x_offset, 4),
                    "y_offset":       round(s.y_offset, 4),
                    "z_offset":       round(s.z_offset, 4),
                    "yaw_deg":        round(s.yaw_deg, 2),
                    "pitch_deg":      round(s.pitch_deg, 2),
                    "range_fraction": round(s.range_fraction, 4),
                    "hfov_fraction":  round(s.hfov_fraction, 4),
                }
                for s in result.best_config.active_sensors()
            ]
        }
        logger.log_final(result.best_loss_result, best_config_dict)
        pareto_summary = [
            {
                "idx": p.index,
                "objectives": p.objectives,
            }
            for p in result.pareto_front
        ]
        with open(logger.run_dir / "pareto_front.json", "w") as f:
            json.dump(pareto_summary, f, indent=2)

    print("\n[Experiment] Done.")
    print(f"[Experiment] Results saved to: results/{result.run_id}/")
    print(f"[Experiment] Best loss: {result.best_loss:.6f}")
    print(f"[Experiment] Pareto points: {len(result.pareto_front)}")
    print(f"[Experiment] Converged: {result.converged}")


if __name__ == "__main__":
    main()