"""
sensor_opt/encoding/config.py

Defines the SensorConfig dataclass and the encode/decode functions that
convert between human-readable configs and the flat float vector that
CMA-ES operates on.

Vector layout (per sensor slot):
  [ type_float, count_float, slot_idx_float,
    x_offset, y_offset, z_offset,
    yaw_deg, pitch_deg,
    range_fraction, hfov_fraction ]
  → 10 floats × N_max_total_slots

type_float is a continuous relaxation of the discrete sensor type:
  0.0 = disabled
  1.0 = lidar
  2.0 = camera
  3.0 = radar
  (snapped to nearest integer at decode time)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

SENSOR_TYPE_MAP = {
    0: "disabled",
    1: "lidar",
    2: "camera",
    3: "radar",
}
SENSOR_TYPE_REVERSE = {v: k for k, v in SENSOR_TYPE_MAP.items()}

FLOATS_PER_SENSOR = 10

PARAM_BOUNDS = {
    "type_float":       (0.0,   3.0),
    "count_float":      (0.0,   1.0),
    "slot_idx_float":   (0.0,   1.0),
    "x_offset":         (-0.5,  0.5),
    "y_offset":         (-0.5,  0.5),
    "z_offset":         (0.0,   0.5),
    "yaw_deg":          (-180.0, 180.0),
    "pitch_deg":        (-45.0,  45.0),
    "range_fraction":   (0.1,   1.0),
    "hfov_fraction":    (0.2,   1.0),
}


@dataclass
class SingleSensorConfig:
    """Configuration for one physical sensor instance."""
    sensor_type: str
    slot: str
    x_offset: float = 0.0
    y_offset: float = 0.0
    z_offset: float = 0.2
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    range_fraction: float = 1.0
    hfov_fraction: float = 1.0

    def is_active(self) -> bool:
        return self.sensor_type != "disabled"


@dataclass
class SensorConfig:
    """Full sensor configuration for one robot."""
    sensors: List[SingleSensorConfig] = field(default_factory=list)

    def active_sensors(self) -> List[SingleSensorConfig]:
        return [s for s in self.sensors if s.is_active()]

    def count_by_type(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for s in self.active_sensors():
            counts[s.sensor_type] = counts.get(s.sensor_type, 0) + 1
        return counts

    def total_cost(self, sensor_models: dict) -> float:
        total = 0.0
        for s in self.active_sensors():
            model = sensor_models.get(s.sensor_type, {})
            total += model.get("cost_usd", 0.0)
        return total

    def summary(self) -> str:
        active = self.active_sensors()
        counts = self.count_by_type()
        parts = [f"{v}×{k}" for k, v in sorted(counts.items())]
        return f"SensorConfig({', '.join(parts) or 'empty'}, total={len(active)} sensors)"


def encode(
    config: SensorConfig,
    mounting_slots: List[str],
    fixed_mount_order: bool = False,
) -> np.ndarray:
    """Encode a SensorConfig into a flat float vector for CMA-ES."""
    n_slots = len(config.sensors)
    vec = np.zeros(n_slots * FLOATS_PER_SENSOR, dtype=np.float64)

    for i, sensor in enumerate(config.sensors):
        base = i * FLOATS_PER_SENSOR
        vec[base + 0] = float(SENSOR_TYPE_REVERSE.get(sensor.sensor_type, 0))
        vec[base + 1] = 1.0 if sensor.is_active() else 0.0
        if fixed_mount_order and i < len(mounting_slots):
            slot_idx = i
        else:
            slot_idx = mounting_slots.index(sensor.slot) if sensor.slot in mounting_slots else 0
        vec[base + 2] = slot_idx / max(len(mounting_slots) - 1, 1)
        vec[base + 3] = sensor.x_offset
        vec[base + 4] = sensor.y_offset
        vec[base + 5] = sensor.z_offset
        vec[base + 6] = sensor.yaw_deg / 180.0
        vec[base + 7] = sensor.pitch_deg / 45.0
        vec[base + 8] = sensor.range_fraction
        vec[base + 9] = sensor.hfov_fraction

    return vec


def decode(
    vec: np.ndarray,
    mounting_slots: List[str],
    sensor_budget: dict,
    snap_discrete: bool = True,
    fixed_mount_order: bool = False,
) -> SensorConfig:
    """Decode a CMA-ES float vector back into a SensorConfig."""
    n_sensors = len(vec) // FLOATS_PER_SENSOR
    sensors: List[SingleSensorConfig] = []
    type_counts: Dict[str, int] = {}

    for i in range(n_sensors):
        base = i * FLOATS_PER_SENSOR
        chunk = vec[base: base + FLOATS_PER_SENSOR].copy()

        raw_type = float(np.clip(chunk[0], 0.0, 3.0))
        type_idx = int(round(raw_type)) if snap_discrete else int(raw_type)
        sensor_type = SENSOR_TYPE_MAP.get(type_idx, "disabled")

        if sensor_type != "disabled":
            max_allowed = sensor_budget.get(sensor_type, {}).get("max_count", 0)
            current = type_counts.get(sensor_type, 0)
            if current >= max_allowed:
                sensor_type = "disabled"

        active_flag = float(np.clip(chunk[1], 0.0, 1.0))
        if snap_discrete and active_flag < 0.5:
            sensor_type = "disabled"

        if fixed_mount_order and i < len(mounting_slots):
            slot = mounting_slots[i]
        else:
            raw_slot = float(np.clip(chunk[2], 0.0, 1.0))
            slot_idx = int(round(raw_slot * (len(mounting_slots) - 1)))
            slot_idx = max(0, min(slot_idx, len(mounting_slots) - 1))
            slot = mounting_slots[slot_idx]

        x_off = float(np.clip(chunk[3], -0.5, 0.5))
        y_off = float(np.clip(chunk[4], -0.5, 0.5))
        z_off = float(np.clip(chunk[5], 0.0, 0.5))
        yaw   = float(np.clip(chunk[6], -1.0, 1.0)) * 180.0
        pitch = float(np.clip(chunk[7], -1.0, 1.0)) * 45.0
        rng_f = float(np.clip(chunk[8], 0.1, 1.0))
        fov_f = float(np.clip(chunk[9], 0.2, 1.0))

        if sensor_type != "disabled":
            type_counts[sensor_type] = type_counts.get(sensor_type, 0) + 1

        sensors.append(SingleSensorConfig(
            sensor_type=sensor_type,
            slot=slot,
            x_offset=x_off,
            y_offset=y_off,
            z_offset=z_off,
            yaw_deg=yaw,
            pitch_deg=pitch,
            range_fraction=rng_f,
            hfov_fraction=fov_f,
        ))

    return SensorConfig(sensors=sensors)


def config_vector_size(sensor_budget: dict) -> int:
    """Return the total CMA-ES vector length for a given sensor budget."""
    total_slots = sum(v.get("max_count", 0) for v in sensor_budget.values())
    return total_slots * FLOATS_PER_SENSOR


def make_initial_vector(sensor_budget: dict, mounting_slots: List[str]) -> np.ndarray:
    """Create a starting vector with all sensors disabled."""
    n = config_vector_size(sensor_budget)
    vec = np.zeros(n, dtype=np.float64)
    n_sensors = n // FLOATS_PER_SENSOR
    for i in range(n_sensors):
        base = i * FLOATS_PER_SENSOR
        vec[base + 0] = 0.0
        vec[base + 1] = 0.0
        vec[base + 2] = 0.5
        vec[base + 5] = 0.2
        vec[base + 8] = 1.0
        vec[base + 9] = 1.0
    return vec