"""Default sensor parameters; optional YAML overrides (merged at runtime)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

_DEFAULT_SENSOR_SPECS: Dict[str, Dict[str, Any]] = {
    "camera": {
        "fov_deg": 60.0,
        "max_range_m": 15.0,
        "latency_s": 0.033,
        "n_rays": 32,
        "detection_conf": 0.85,
    },
    "radar": {
        "fov_deg": 90.0,
        "max_range_m": 30.0,
        "latency_s": 0.05,
        "n_rays": 16,
        "detection_conf": 0.70,
    },
    "lidar": {
        "fov_deg": 120.0,
        "max_range_m": 25.0,
        "latency_s": 0.10,
        "n_rays": 64,
        "detection_conf": 0.95,
    },
    "disabled": {
        "fov_deg": 0.0,
        "max_range_m": 0.0,
        "latency_s": None,
        "n_rays": 0,
        "detection_conf": 0.0,
    },
}


def merge_sensor_spec_overrides(overrides: Dict[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    """Deep-copy defaults and update from optional YAML / runtime dict."""
    out = deepcopy(_DEFAULT_SENSOR_SPECS)
    if not overrides:
        return out
    for name, ovr in overrides.items():
        if name not in out or not isinstance(ovr, dict):
            continue
        for k, v in ovr.items():
            if v is not None:
                out[name][k] = v
    if "disabled" in overrides and isinstance(overrides["disabled"], dict):
        d = out["disabled"]
        for k, v in overrides["disabled"].items():
            if v is not None:
                d[k] = v
    return out


def get_sensor_specs(experiment_config: dict | None) -> Dict[str, Dict[str, Any]]:
    """Read `inner_loop.mujoco.sensor_spec_overrides` if present (YAML or runtime)."""
    if not experiment_config:
        return deepcopy(_DEFAULT_SENSOR_SPECS)
    ovr: Dict[str, Any] | None = None
    il = experiment_config.get("inner_loop")
    if isinstance(il, dict):
        mj = il.get("mujoco")
        if isinstance(mj, dict):
            ovr = mj.get("sensor_spec_overrides")
    sim = experiment_config.get("simulation")
    if ovr is None and isinstance(sim, dict):
        ovr = sim.get("sensor_spec_overrides")
    if not isinstance(ovr, dict):
        ovr = {}
    return merge_sensor_spec_overrides(ovr)


SENSOR_SPECS = _DEFAULT_SENSOR_SPECS
