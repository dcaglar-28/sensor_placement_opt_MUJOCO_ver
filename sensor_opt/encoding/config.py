"""
sensor_opt/encoding/config.py

Defines the SensorConfig dataclass and the encode/decode functions that
convert between human-readable configs and the flat float vector that
CMA-ES operates on.

Vector layout (per sensor slot), full mode (`fixed_sensor_geometry: false`):
  [ type_float, count_float, slot_idx_float,
    x_offset, y_offset, z_offset,
    yaw_deg, pitch_deg,
    range_fraction, hfov_fraction ]
  → 10 floats × N_max_total_slots

When `fixed_sensor_geometry: true` (uses fixed mount order: slot i = block i):
  [ type_float, active_float ] only → 2 floats × N_max_total_slots
  x/y/z/yaw/pitch/range/fov come from `default_sensor_pose` in the experiment config.
  The loss, `hardware:` penalties, and `sensor_models` cost use the same `compute_loss` path; only
  the CMA-ES search dimension (what is optimized) changes, not the objective or YAML hardware block.

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
from typing import Any, Dict, List, Optional

import numpy as np

SENSOR_TYPE_MAP = {
    0: "disabled",
    1: "lidar",
    2: "camera",
    3: "radar",
}
SENSOR_TYPE_REVERSE = {v: k for k, v in SENSOR_TYPE_MAP.items()}

FLOATS_PER_SENSOR = 10  # full layout; use `floats_per_sensor(fixed_sensor_geometry=False)` in new code
FLOATS_PER_SENSOR_FULL = 10
FLOATS_PER_SENSOR_ALLOC = 2
FLOATS_PER_SENSOR_VEHICLE5 = 1  # [0,1] bucketed type per fixed slot; no pose offsets

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

_POSE_KEYS = (
    "x_offset",
    "y_offset",
    "z_offset",
    "yaw_deg",
    "pitch_deg",
    "range_fraction",
    "hfov_fraction",
)


def floats_per_sensor(
    fixed_sensor_geometry: bool,
    vehicle_5slot: bool = False,
) -> int:
    """1 when 5 fixed slots / [0,1] type genes; 2 for type+active; 10 for full layout."""
    if vehicle_5slot:
        return FLOATS_PER_SENSOR_VEHICLE5
    return FLOATS_PER_SENSOR_ALLOC if fixed_sensor_geometry else FLOATS_PER_SENSOR_FULL


def _gene_to_type_01(x: float) -> str:
    """[0,0.25) disabled, [0.25,0.5) camera, [0.5,0.75) radar, [0.75,1] lidar."""
    c = float(np.clip(x, 0.0, 1.0))
    if c < 0.25:
        return "disabled"
    if c < 0.5:
        return "camera"
    if c < 0.75:
        return "radar"
    return "lidar"


def _type_to_gene_01(sensor_type: str) -> float:
    mid = {"disabled": 0.125, "camera": 0.375, "radar": 0.625, "lidar": 0.875}
    return float(mid.get(sensor_type, 0.125))


def merge_default_sensor_pose(
    sensor_type: str,
    slot: str,
    default_sensor_pose: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Build x/y/z/yaw/pitch/range/fov from `default_sensor_pose` (YAML).
    Merges `all` → per-type (lidar, camera, radar) → per_slot[slot] when present.
    Intentional defaults: neutral forward boresight (0° yaw, 0° pitch) and a small z lift.
    """
    base: Dict[str, float] = {
        "x_offset": 0.0,
        "y_offset": 0.0,
        "z_offset": 0.2,
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "range_fraction": 1.0,
        "hfov_fraction": 1.0,
    }
    d = default_sensor_pose or {}
    for section in (d.get("all"), d.get(sensor_type), (d.get("per_slot") or {}).get(slot)):
        if not isinstance(section, dict):
            continue
        for k in _POSE_KEYS:
            if k in section:
                base[k] = float(section[k])
    return base


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


def reapply_default_geometry(
    config: SensorConfig,
    default_sensor_pose: Optional[Dict[str, Any]],
    fixed_sensor_geometry: bool,
) -> SensorConfig:
    """
    For NSGA-2 (and similar) that mutate `SensorConfig` in place: reset offsets/orientation
    to the YAML defaults so allocation-only search does not drift pose between generations.
    """
    if not fixed_sensor_geometry:
        return config
    out: List[SingleSensorConfig] = []
    for s in config.sensors:
        pose = merge_default_sensor_pose(s.sensor_type, s.slot, default_sensor_pose)
        if s.sensor_type == "disabled":
            out.append(
                SingleSensorConfig(
                    sensor_type="disabled",
                    slot=s.slot,
                    x_offset=pose["x_offset"],
                    y_offset=pose["y_offset"],
                    z_offset=pose["z_offset"],
                    yaw_deg=pose["yaw_deg"],
                    pitch_deg=pose["pitch_deg"],
                    range_fraction=pose["range_fraction"],
                    hfov_fraction=pose["hfov_fraction"],
                )
            )
        else:
            out.append(
                SingleSensorConfig(
                    sensor_type=s.sensor_type,
                    slot=s.slot,
                    x_offset=pose["x_offset"],
                    y_offset=pose["y_offset"],
                    z_offset=pose["z_offset"],
                    yaw_deg=pose["yaw_deg"],
                    pitch_deg=pose["pitch_deg"],
                    range_fraction=pose["range_fraction"],
                    hfov_fraction=pose["hfov_fraction"],
                )
            )
    return SensorConfig(sensors=out)


def encode(
    config: SensorConfig,
    mounting_slots: List[str],
    fixed_mount_order: bool = False,
    fixed_sensor_geometry: bool = False,
    vehicle_5slot: bool = False,
) -> np.ndarray:
    """Encode a SensorConfig into a flat float vector for CMA-ES (or 2-dim / slot in allocation mode)."""
    fper = floats_per_sensor(fixed_sensor_geometry, vehicle_5slot=vehicle_5slot)
    if vehicle_5slot:
        v = np.zeros(5, dtype=np.float64)
        for i, slot in enumerate(mounting_slots):
            s = next((x for x in config.sensors if x.slot == slot), None)
            st = s.sensor_type if s is not None else "disabled"
            v[i] = _type_to_gene_01(st)
        return v
    n_slots = len(config.sensors)
    vec = np.zeros(n_slots * fper, dtype=np.float64)

    for i, sensor in enumerate(config.sensors):
        base = i * fper
        vec[base + 0] = float(SENSOR_TYPE_REVERSE.get(sensor.sensor_type, 0))
        vec[base + 1] = 1.0 if sensor.is_active() else 0.0
        if fixed_sensor_geometry:
            continue
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


def _apply_max_sensor_count(config: SensorConfig, mounting_slots: List[str], max_n: Optional[int]) -> SensorConfig:
    if max_n is None or max_n < 0:
        return config
    act = [s for s in config.sensors if s.is_active()]
    if len(act) <= max_n:
        return config
    # Disable from the end of the fixed mount list first
    to_disable = len(act) - max_n
    disabled_slots = set()
    for slot in reversed(mounting_slots):
        if to_disable <= 0:
            break
        s = next((x for x in config.sensors if x.slot == slot and x.is_active()), None)
        if s is not None:
            disabled_slots.add(slot)
            to_disable -= 1
    new_s: List[SingleSensorConfig] = []
    for s in config.sensors:
        if s.slot in disabled_slots:
            new_s.append(
                SingleSensorConfig(
                    sensor_type="disabled",
                    slot=s.slot,
                    x_offset=0.0,
                    y_offset=0.0,
                    z_offset=0.2,
                    yaw_deg=0.0,
                    pitch_deg=0.0,
                    range_fraction=1.0,
                    hfov_fraction=1.0,
                )
            )
        else:
            new_s.append(s)
    return SensorConfig(sensors=new_s)


def decode(
    vec: np.ndarray,
    mounting_slots: List[str],
    sensor_budget: dict,
    snap_discrete: bool = True,
    fixed_mount_order: bool = False,
    fixed_sensor_geometry: bool = False,
    default_sensor_pose: Optional[Dict[str, Any]] = None,
    vehicle_5slot: bool = False,
    max_sensor_count: Optional[int] = None,
) -> SensorConfig:
    """Decode a CMA-ES float vector back into a SensorConfig."""
    fper = floats_per_sensor(fixed_sensor_geometry, vehicle_5slot=vehicle_5slot)
    use_fixed_mount = bool(fixed_mount_order or fixed_sensor_geometry or vehicle_5slot)
    if vehicle_5slot and len(mounting_slots) != 5:
        raise ValueError("vehicle_5slot requires exactly 5 mounting_slots")
    if vehicle_5slot and int(vec.size) != 5:
        raise ValueError(f"vehicle_5slot requires vector length 5, got {vec.size}")
    if vehicle_5slot:
        sensors: List[SingleSensorConfig] = []
        type_counts: Dict[str, int] = {}
        for i, slot in enumerate(mounting_slots):
            g = float(np.clip(vec[i], 0.0, 1.0))
            sensor_type = _gene_to_type_01(g)
            if sensor_type != "disabled":
                max_allowed = int(sensor_budget.get(sensor_type, {}).get("max_count", 0) or 0)
                if type_counts.get(sensor_type, 0) >= max_allowed:
                    sensor_type = "disabled"
            if sensor_type != "disabled":
                type_counts[sensor_type] = type_counts.get(sensor_type, 0) + 1
            pose = merge_default_sensor_pose(sensor_type, slot, default_sensor_pose)
            sensors.append(
                SingleSensorConfig(
                    sensor_type=sensor_type,
                    slot=slot,
                    x_offset=0.0,
                    y_offset=0.0,
                    z_offset=float(pose["z_offset"]),
                    yaw_deg=0.0,
                    pitch_deg=0.0,
                    range_fraction=1.0,
                    hfov_fraction=1.0,
                )
            )
        cfg = SensorConfig(sensors=sensors)
        return _apply_max_sensor_count(cfg, mounting_slots, max_sensor_count)
    if int(vec.size) % fper != 0:
        raise ValueError(
            f"vector length {vec.size} is not a multiple of {fper} floats per slot"
        )
    # Allow len(vec) != `config_vector_size` for tests and legacy (extra or fewer slot-rows);
    # the outer loop and search still use `config_vector_size` for CMA-ES and stay consistent.
    n_sensors = int(len(vec)) // fper
    sensors: List[SingleSensorConfig] = []
    type_counts: Dict[str, int] = {}

    for i in range(n_sensors):
        base = i * fper
        chunk = np.asarray(vec[base: base + fper], dtype=np.float64).copy()
        if chunk.size < 2:
            raise ValueError("decode: need at least 2 values per slot")

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

        if use_fixed_mount and i < len(mounting_slots):
            slot = mounting_slots[i]
        elif fper == FLOATS_PER_SENSOR_FULL and len(chunk) > 2:
            raw_slot = float(np.clip(chunk[2], 0.0, 1.0))
            slot_idx = int(round(raw_slot * (len(mounting_slots) - 1)))
            slot_idx = max(0, min(slot_idx, len(mounting_slots) - 1))
            slot = mounting_slots[slot_idx]
        else:
            slot = mounting_slots[min(i, len(mounting_slots) - 1)] if mounting_slots else "front"

        if fixed_sensor_geometry:
            pose = merge_default_sensor_pose(sensor_type, slot, default_sensor_pose)
            x_off = float(pose["x_offset"])
            y_off = float(pose["y_offset"])
            z_off = float(pose["z_offset"])
            yaw = float(pose["yaw_deg"])
            pitch = float(pose["pitch_deg"])
            rng_f = float(pose["range_fraction"])
            fov_f = float(pose["hfov_fraction"])
        else:
            x_off = float(np.clip(chunk[3], -0.5, 0.5))
            y_off = float(np.clip(chunk[4], -0.5, 0.5))
            z_off = float(np.clip(chunk[5], 0.0, 0.5))
            yaw = float(np.clip(chunk[6], -1.0, 1.0)) * 180.0
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


def config_vector_size(
    sensor_budget: dict, fixed_sensor_geometry: bool = False, vehicle_5slot: bool = False
) -> int:
    """Return the total CMA-ES vector length for a given sensor budget."""
    if vehicle_5slot:
        return 5
    total_slots = sum(v.get("max_count", 0) for v in sensor_budget.values())
    return total_slots * floats_per_sensor(fixed_sensor_geometry, vehicle_5slot=vehicle_5slot)


def make_initial_vector(
    sensor_budget: dict,
    mounting_slots: List[str],
    fixed_sensor_geometry: bool = False,
    vehicle_5slot: bool = False,
) -> np.ndarray:
    """
    CMA-ES start vector: **must not** leave all slots disabled, or the search stays at loss = 1.0.

    For `fixed_sensor_geometry` (2 floats / slot: type, active), we set a diverse mix of
    lidar / camera / radar (types 1–3) with active > 0.5 so the decode path turns sensors ON.
    For full 10-float mode, we also enable sensors and keep neutral pose hints.
    """
    fper = floats_per_sensor(fixed_sensor_geometry, vehicle_5slot=vehicle_5slot)
    n = config_vector_size(sensor_budget, fixed_sensor_geometry, vehicle_5slot=vehicle_5slot)
    vec = np.zeros(n, dtype=np.float64)
    n_sensors = n // fper
    for i in range(n_sensors):
        base = i * fper
        if vehicle_5slot:
            vec[base + 0] = _type_to_gene_01(["camera", "radar", "lidar", "radar", "camera"][i % 5])
        elif fixed_sensor_geometry:
            # type_float in (1,2,3) = lidar, camera, radar; active_float high so not snapped off
            vec[base + 0] = float(1.0 + (i % 3))  # 1.0, 2.0, 3.0 cyclically
            vec[base + 1] = 0.85
        else:
            vec[base + 0] = 1.0 + float((i % 3))
            vec[base + 1] = 0.85
            vec[base + 2] = 0.5
            vec[base + 5] = 0.2
            vec[base + 8] = 1.0
            vec[base + 9] = 1.0
    return vec