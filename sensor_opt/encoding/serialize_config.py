"""Serialize `SensorConfig` for JSON logs and paper figures."""

from __future__ import annotations

from sensor_opt.encoding.config import SensorConfig, SingleSensorConfig


def single_sensor_to_dict(s: SingleSensorConfig) -> dict:
    return {
        "type": s.sensor_type,
        "slot": s.slot,
        "x_offset": float(s.x_offset),
        "y_offset": float(s.y_offset),
        "z_offset": float(s.z_offset),
        "yaw_deg": float(s.yaw_deg),
        "pitch_deg": float(s.pitch_deg),
        "range_fraction": float(s.range_fraction),
        "hfov_fraction": float(s.hfov_fraction),
    }


def sensor_config_to_dict(config: SensorConfig) -> dict:
    return {"sensors": [single_sensor_to_dict(s) for s in config.sensors]}
