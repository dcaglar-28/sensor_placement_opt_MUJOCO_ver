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

import jax  # noqa: F401
import matplotlib  # noqa: F401
import mlflow  # noqa: F401
import pandas  # noqa: F401
import rich  # noqa: F401
import scipy  # noqa: F401
import sklearn  # noqa: F401
import torch  # noqa: F401

from sensor_opt.config.specs import prepare_experiment_config
from sensor_opt.evaluation.pipeline import Evaluator
from sensor_opt.inner_loop.isaac_evaluator import IsaacSimEvaluator
from sensor_opt.inner_loop.mock_isaac_evaluator import MockIsaacEvaluator
from sensor_opt.logging.experiment_logger import ExperimentLogger
from sensor_opt.loss.loss import loss_weight_dict
from sensor_opt.search.factory import create_search
from sensor_opt.simulation.mjcf import SLOT_NAMES


def load_config(path: str, runtime_overrides: dict | None = None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return prepare_experiment_config(cfg, runtime_overrides=runtime_overrides)


def main():
    parser = argparse.ArgumentParser(description="Sensor Placement Optimizer")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--dummy", action="store_true",
                        help="Alias for mock inner-loop evaluator (fast analytic metrics, no physics)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed override")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow logging")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("search", {"type": "cma"})

    if args.dummy:
        cfg["inner_loop"]["mode"] = "mock_isaac"

    rt0 = cfg.get("runtime", {})
    if isinstance(rt0, dict) and rt0.get("trial_type"):
        tt = str(rt0["trial_type"]).lower()
        mname = f"mujoco_{tt}" if not tt.startswith("mujoco_") else tt
        cfg.setdefault("experiment", {})["name"] = mname
        tm = {
            "accuracy": "trial_accuracy",
            "speed": "trial_speed",
            "cost": "trial_cost",
            "multi_objective": "trial_multi_objective",
            "multi": "trial_multi_objective",
        }.get(tt, "trial_multi_objective")
        cfg.setdefault("loss", {})["mode"] = tm
        for k, yaml_key in (
            ("PATH_LENGTH_M", "path_length_m"),
            ("VEHICLE_SPEED_MPS", "vehicle_speed_mps"),
            ("N_OBSTACLES", "n_obstacles"),
            ("BASE_RANDOM_SEED", "base_random_seed"),
        ):
            if k in rt0 and rt0[k] is not None:
                cfg.setdefault("inner_loop", {}).setdefault("mujoco", {})[yaml_key] = rt0[k]
        if "MAX_HARDWARE_BUDGET_USD" in rt0 and rt0["MAX_HARDWARE_BUDGET_USD"] is not None:
            cfg["loss"]["max_hardware_budget_usd"] = float(rt0["MAX_HARDWARE_BUDGET_USD"])
            cfg["loss"]["max_cost_usd"] = float(rt0["MAX_HARDWARE_BUDGET_USD"])
        if "SENSOR_COSTS_USD" in rt0 and isinstance(rt0["SENSOR_COSTS_USD"], dict):
            for tname, c in rt0["SENSOR_COSTS_USD"].items():
                if tname in (cfg.get("sensor_models") or {}):
                    cfg["sensor_models"][tname]["cost_usd"] = float(c)
        if rt0.get("MAX_SENSOR_COUNT") is not None:
            cfg["max_sensor_count"] = rt0["MAX_SENSOR_COUNT"]
        if "LOSS_WEIGHT_OVERRIDES" in rt0 and isinstance(rt0["LOSS_WEIGHT_OVERRIDES"], dict):
            cfg["loss"]["trial_weight_overrides"] = dict(rt0["LOSS_WEIGHT_OVERRIDES"])

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
        print("[Experiment] Using mock inner-loop evaluator")
    elif mode == "isaac_sim":
        base_evaluator = IsaacSimEvaluator(isaac_sim_cfg=cfg["inner_loop"].get("isaac_sim", {}))
        evaluator_fn = None
        print("[Experiment] Using external sim backend (custom env in config)")
    elif mode == "mujoco":
        from sensor_opt.inner_loop.mujoco_evaluator import MujocoSimEvaluator

        mj = dict(cfg["inner_loop"].get("mujoco", {}) or {})
        mj.setdefault(
            "sensor_noise_std",
            float(cfg["inner_loop"].get("isaac_sim", {}).get("sensor_noise_std", 0.0) or 0.0),
        )
        mj.setdefault("max_steps_per_episode", int(cfg["inner_loop"].get("max_steps_per_episode", 500)))
        base_evaluator = MujocoSimEvaluator(
            mujoco_cfg={**mj, "experiment_config": cfg, "n_obstacles": int(mj.get("n_obstacles", 10))}
        )
        evaluator_fn = None
        print("[Experiment] Using MuJoCo evaluator")
    else:
        print(
            f"[Experiment] Unknown mode '{mode}'. "
            "Set inner_loop.mode in configs/default.yaml to dummy/mujoco or the sim branch described there."
        )
        sys.exit(1)

    mf_cfg = cfg.get("multi_fidelity", {})
    if mf_cfg.get("enabled", False):
        evaluator_obj = Evaluator(
            # Use the mock evaluator with zero latency as
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

    exp_name = str(cfg.get("experiment", {}).get("name", "mujoco_run"))
    with ExperimentLogger(
        experiment_name=exp_name,
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

        if bool((cfg.get("encoding") or {}).get("vehicle_5slot")):
            sm = cfg.get("sensor_models", {})
            best_config_dict = {
                "sensors": [
                    {
                        "type": s.sensor_type,
                        "slot": s.slot,
                        "cost_usd": int(round(float(sm.get(s.sensor_type, {}).get("cost_usd", 0) or 0))),
                    }
                    for s in result.best_config.sensors
                    if s.slot in SLOT_NAMES
                ]
            }
        else:
            best_config_dict = {
                "sensors": [
                    {
                        "type": s.sensor_type,
                        "slot": s.slot,
                        "x_offset": round(s.x_offset, 4),
                        "y_offset": round(s.y_offset, 4),
                        "z_offset": round(s.z_offset, 4),
                        "yaw_deg": round(s.yaw_deg, 2),
                        "pitch_deg": round(s.pitch_deg, 2),
                        "range_fraction": round(s.range_fraction, 4),
                        "hfov_fraction": round(s.hfov_fraction, 4),
                    }
                    for s in result.best_config.active_sensors()
                ]
            }
        logger.log_final(result.best_loss_result, best_config_dict)
        # Pareto + full evaluated pool: written in `cma/outer_loop.py` as pareto_front.json / evaluated_pool.json

    _print_experiment_box(cfg, result)
    print("\n[Experiment] Done.")
    print(f"[Experiment] Results saved to: results/{result.run_id}/")
    print(f"[Experiment] Best loss: {result.best_loss:.6f}")
    print(f"[Experiment] Pareto points: {len(result.pareto_front)}")
    print(f"[Experiment] Converged: {result.converged}")


def _print_experiment_box(cfg: dict, result) -> None:
    if not bool((cfg.get("encoding") or {}).get("vehicle_5slot")):
        return
    from sensor_opt.objectives.trial_objectives import trial_display_metrics

    lm = str((cfg.get("loss") or {}).get("mode", ""))
    if "trial_" not in lm:
        return
    tt = (cfg.get("runtime") or {}).get("trial_type") or lm.replace("trial_", "")
    o = (result.best_loss_result.objectives) or {}
    sm = cfg.get("sensor_models", {})
    print("\n  ╔══════════════════════════════════════════════════╗")
    print(f"  ║  TRIAL        : {str(tt):<32}  ║")
    print(f"  ║  Generations  : {result.n_generations:<32}  ║")
    print(f"  ║  Best loss    : {result.best_loss:.3f}{' ' * 23}  ║")
    print("  ╠══════════════════════════════════════════════════╣")
    print("  ║  SENSOR LAYOUT                                   ║")
    for s in result.best_config.sensors:
        if s.slot in SLOT_NAMES:
            cst = int(round(float(sm.get(s.sensor_type, {}).get("cost_usd", 0) or 0)))
            print(f"  ║    {s.slot:<20} → {s.sensor_type:<8} (${cst}){(' ' * max(0, 18 - len(s.sensor_type)))}  ║")
    na = result.best_loss_result.n_active_sensors
    bmax = float((cfg.get("loss") or {}).get("max_hardware_budget_usd", 5000) or 5000)
    tc = result.best_loss_result.cost_usd
    dm = trial_display_metrics(result.best_loss_result)
    print("  ╠══════════════════════════════════════════════════╣")
    print(f"  ║  Active sensors : {na} / 5 slots{' ' * 20}  ║")
    print(f"  ║  Total cost     : ${tc:,.0f} / ${bmax:,.0f}{' ' * 11}  ║")
    print(f"  ║  Obstacles det. : {str(dm.get('obstacles_detected', 'n/a')):<32}  ║")
    print(f"  ║  Avg detection  : {str(dm.get('mean_detection_m', 'n/a')):<32}  ║")
    print(f"  ║  Coverage frac. : {o.get('coverage_fraction', 0.0):.2f}{' ' * 27}  ║")
    print("  ╚══════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()