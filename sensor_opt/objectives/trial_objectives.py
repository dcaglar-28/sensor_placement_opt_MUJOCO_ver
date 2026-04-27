"""
Trial losses: accuracy, speed, multi-objective (see project architecture doc).
All components normalized to [0,1] before weighting.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.loss.loss import EvalMetrics, LossResult
from sensor_opt.simulation.sensor_specs import merge_sensor_spec_overrides

MAX_TRIAL_LOSS = 1e6

_TR_DEFAULT: Dict[str, Dict[str, float]] = {
    "accuracy": {"w_det": 0.4, "w_dist": 0.2, "w_conf": 0.2, "w_cov": 0.2},
    "speed": {"w_lat": 0.4, "w_time": 0.4, "w_cov": 0.2},
    "multi_objective": {"w_acc": 0.4, "w_lat": 0.4, "w_cost": 0.2},
    # Cost-only: minimize hardware cost fraction (normalized by max budget).
    "cost": {"w_cost": 1.0},
}


def default_trial_weight_overrides(trial_type: str) -> Dict[str, float]:
    t = str(trial_type).lower()
    if t == "accuracy":
        return dict(_TR_DEFAULT["accuracy"])
    if t == "speed":
        return dict(_TR_DEFAULT["speed"])
    if t == "cost":
        return dict(_TR_DEFAULT["cost"])
    if t in ("multi_objective", "multi"):
        return dict(_TR_DEFAULT["multi_objective"])
    return {}


def _merge_weights(trial_type: str, ovr: Optional[Dict[str, float]]) -> Dict[str, float]:
    w = default_trial_weight_overrides(trial_type)
    if ovr:
        w = {**w, **{k: float(v) for k, v in ovr.items() if v is not None}}
    return w


def _plain_cost_usd(config: SensorConfig, sensor_models: dict) -> float:
    t = 0.0
    for s in config.active_sensors():
        t += float(sensor_models.get(s.sensor_type, {}).get("cost_usd", 0.0) or 0.0)
    return t


def detection_confidence_term(
    config: SensorConfig, sensor_specs: Dict[str, Dict[str, Any]], per_slot_hits: Optional[Dict[str, int]]
) -> float:
    """L_conf = 1 - weighted mean(detection_conf) with weights = first-hit counts per active slot."""
    if not per_slot_hits:
        return 1.0
    num, den = 0.0, 0.0
    for s in config.sensors:
        if not s.is_active():
            continue
        conf = float((sensor_specs.get(s.sensor_type) or {}).get("detection_conf", 0.0) or 0.0)
        hits = int(per_slot_hits.get(s.slot, 0) or 0)
        if hits <= 0:
            continue
        num += conf * hits
        den += float(hits)
    if den < 1e-9:
        return 1.0
    return 1.0 - float(num / den)


def _active_max_range_m(sensor_specs: Dict[str, Dict[str, Any]], config: SensorConfig) -> float:
    rs = [
        float((sensor_specs.get(s.sensor_type) or {}).get("max_range_m", 0.0) or 0.0)
        for s in config.active_sensors()
    ]
    r = max(rs) if rs else 0.0
    return r if r > 1e-6 else 30.0


def compute_trial_loss(
    *,
    trial_type: str,
    metrics: EvalMetrics,
    config: SensorConfig,
    sensor_models: dict,
    trial_weights: Optional[Dict[str, float]] = None,
    max_hardware_budget_usd: float = 10_000.0,
    sensor_specs: Optional[Dict[str, Dict[str, Any]]] = None,
) -> LossResult:
    t = str(trial_type).lower()
    sspec = sensor_specs or merge_sensor_spec_overrides({})
    wts = _merge_weights(t, trial_weights)
    n_act = len(config.active_sensors())
    if n_act == 0:
        return _huge(t, config, sspec)

    n_obs = int(getattr(metrics, "n_obstacles", 0) or 0)
    n_det = int(getattr(metrics, "n_detected", 0) or 0)
    if n_obs <= 0:
        n_obs = 1
    l_det = float(np.clip(1.0 - float(n_det) / float(n_obs), 0.0, 1.0))

    mdd = float(getattr(metrics, "mean_detection_distance_m", 0.0) or 0.0)
    m_r = _active_max_range_m(sspec, config)
    l_dist = 1.0 - float(mdd) / m_r
    l_dist = float(np.clip(l_dist, 0.0, 1.0))

    cov = float(getattr(metrics, "coverage_fraction", 0.0) or 0.0)
    l_cov = 1.0 - float(np.clip(cov, 0.0, 1.0))

    pslot = getattr(metrics, "per_slot_first_hits", None)
    pslot = pslot if isinstance(pslot, dict) else {}
    l_conf = float(
        np.clip(detection_confidence_term(config, sspec, pslot), 0.0, 1.0)
    )

    ep_dur = float(getattr(metrics, "episode_time_s", 0.0) or 0.0)
    if ep_dur < 1e-9:
        ep_dur = 1.0
    ftm = float(getattr(metrics, "first_detection_time_mean", 0.0) or 0.0)
    l_time = float(np.clip(ftm / ep_dur, 0.0, 1.0))

    lat_m = 0.1
    lats = [
        float((sspec.get(s.sensor_type) or {}).get("latency_s", 0.0) or 0.0) for s in config.active_sensors()
    ]
    mean_lat = float(np.mean(lats)) if lats else lat_m
    l_lat = float(np.clip(mean_lat / lat_m, 0.0, 1.0))

    cost_usd = _plain_cost_usd(config, sensor_models)
    l_cost = float(np.clip(cost_usd / max(max_hardware_budget_usd, 1e-6), 0.0, 1.0))

    term_acc = 0.0
    term_speed = 0.0
    term_cost = 0.0
    term_cov = 0.0

    if t == "accuracy":
        w = wts
        total = (
            w.get("w_det", 0.4) * l_det
            + w.get("w_dist", 0.2) * l_dist
            + w.get("w_conf", 0.2) * l_conf
            + w.get("w_cov", 0.2) * l_cov
        )
        term_acc = w.get("w_det", 0.4) * l_det + w.get("w_dist", 0.2) * l_dist + w.get("w_conf", 0.2) * l_conf
        term_cov = w.get("w_cov", 0.2) * l_cov
    elif t == "speed":
        w = wts
        total = (
            w.get("w_lat", 0.4) * l_lat
            + w.get("w_time", 0.4) * l_time
            + w.get("w_cov", 0.2) * l_cov
        )
        term_speed = w.get("w_lat", 0.4) * l_lat + w.get("w_time", 0.4) * l_time
        term_cov = w.get("w_cov", 0.2) * l_cov
    elif t in ("multi_objective", "multi"):
        w = wts
        term_acc = w.get("w_acc", 0.4) * l_det
        term_speed = w.get("w_lat", 0.4) * l_time
        term_cost = w.get("w_cost", 0.2) * l_cost
        if cost_usd > max_hardware_budget_usd + 1e-6:
            total = MAX_TRIAL_LOSS
        else:
            total = term_acc + term_speed + term_cost
    elif t == "cost":
        w = wts
        term_cost = w.get("w_cost", 1.0) * l_cost
        total = term_cost
    else:
        total = 1.0

    objectives = {
        "collision": float(l_det),
        "blind_spot": float(l_time),
        "cost": float(l_cost),
        "hardware": 0.0,
        "coverage": float(1.0 - l_cov),
        "L_detection_rate": float(l_det),
        "L_first_detection_time": float(l_time),
        "L_coverage": float(l_cov),
        "L_hardware": float(l_cost),
        "term_group_accuracy": float(term_acc),
        "term_group_speed": float(term_speed),
        "term_group_cost": float(term_cost),
        "term_group_coverage": float(term_cov),
        "mean_detection_distance_m": float(mdd),
        "coverage_fraction": float(cov),
        "n_detected": float(n_det),
        "n_obstacles": float(n_obs),
        "cost_usd": float(cost_usd),
    }

    return LossResult(
        total=float(total),
        collision_term=l_det,
        blind_term=l_time,
        cost_term=term_cost if t in ("multi_objective", "multi", "cost") else 0.0,
        cost_usd=cost_usd,
        n_active_sensors=n_act,
        config_summary=_layout_summary(config, sensor_models),
        hardware_penalty_term=0.0,
        objectives=objectives,
    )


def _huge(t: str, config: SensorConfig, sspec: Dict[str, Dict[str, Any]]) -> LossResult:
    _ = t, sspec
    o = {k: 0.0 for k in (
        "collision", "blind_spot", "cost", "hardware", "coverage", "L_detection_rate", "L_first_detection_time",
        "L_coverage", "L_hardware", "term_group_accuracy", "term_group_speed", "term_group_cost", "term_group_coverage",
        "mean_detection_distance_m", "coverage_fraction", "n_detected", "n_obstacles", "cost_usd"
    )}
    o["collision"] = 1.0
    o["blind_spot"] = 1.0
    return LossResult(
        total=MAX_TRIAL_LOSS,
        collision_term=1.0,
        blind_term=1.0,
        cost_term=0.0,
        cost_usd=0.0,
        n_active_sensors=0,
        config_summary=config.summary(),
        hardware_penalty_term=0.0,
        objectives=o,
    )


def _layout_summary(config: SensorConfig, sensor_models: dict) -> str:
    parts: List[str] = []
    for s in config.sensors:
        c = int(sensor_models.get(s.sensor_type, {}).get("cost_usd", 0) or 0)
        parts.append(f"{s.slot}={s.sensor_type}(${c})")
    return "; ".join(parts) if parts else "empty"


def trial_display_metrics(loss: LossResult) -> Dict[str, str]:
    o = loss.objectives or {}
    n_o = int(o.get("n_obstacles", 0) or 0)
    return {
        "obstacles_detected": f"{o.get('n_detected', 0):.1f} / {n_o} avg per episode",
        "mean_detection_m": f"{o.get('mean_detection_distance_m', 0.0):.1f}m from vehicle",
        "coverage": f"{o.get('coverage_fraction', 0.0):.2f}",
    }
