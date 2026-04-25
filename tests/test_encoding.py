"""
tests/test_encoding.py
Unit tests for encode/decode round-trip correctness.
Run with: pytest tests/ -v
"""

import numpy as np
import pytest

from sensor_opt.encoding.config import (
    FLOATS_PER_SENSOR,
    SingleSensorConfig,
    SensorConfig,
    config_vector_size,
    decode,
    encode,
    floats_per_sensor,
    make_initial_vector,
    merge_default_sensor_pose,
)

SLOTS = ["front", "rear", "left", "right", "top", "front-left", "front-right"]

BUDGET = {
    "lidar":  {"max_count": 2},
    "camera": {"max_count": 2},
    "radar":  {"max_count": 1},
}

BUDGET_SMALL = {
    "lidar":  {"max_count": 1},
    "camera": {"max_count": 1},
}


def make_two_sensor_config() -> SensorConfig:
    return SensorConfig(sensors=[
        SingleSensorConfig("lidar",    "top",   z_offset=0.3, yaw_deg=0.0,   range_fraction=1.0, hfov_fraction=1.0),
        SingleSensorConfig("camera",   "front", z_offset=0.2, yaw_deg=10.0,  range_fraction=0.8, hfov_fraction=0.9),
        SingleSensorConfig("disabled", "rear"),
        SingleSensorConfig("radar",    "rear",  z_offset=0.1, yaw_deg=180.0, range_fraction=0.7, hfov_fraction=0.6),
        SingleSensorConfig("disabled", "left"),
    ])


def test_config_vector_size_matches_budget():
    total_slots = sum(v["max_count"] for v in BUDGET.values())
    assert config_vector_size(BUDGET) == total_slots * FLOATS_PER_SENSOR


def test_config_vector_size_small():
    assert config_vector_size(BUDGET_SMALL) == 2 * FLOATS_PER_SENSOR


def test_make_initial_vector_shape():
    vec = make_initial_vector(BUDGET, SLOTS)
    assert vec.shape == (config_vector_size(BUDGET),)


def test_make_initial_vector_all_disabled():
    vec = make_initial_vector(BUDGET, SLOTS)
    n = config_vector_size(BUDGET) // FLOATS_PER_SENSOR
    for i in range(n):
        base = i * FLOATS_PER_SENSOR
        assert vec[base + 0] == 0.0
        assert vec[base + 1] == 0.0


def test_encode_returns_correct_shape():
    cfg = make_two_sensor_config()
    vec = encode(cfg, SLOTS)
    assert vec.shape == (len(cfg.sensors) * FLOATS_PER_SENSOR,)


def test_encode_dtype_is_float64():
    cfg = make_two_sensor_config()
    vec = encode(cfg, SLOTS)
    assert vec.dtype == np.float64


def test_decode_disabled_sensors_are_disabled():
    vec = make_initial_vector(BUDGET, SLOTS)
    cfg = decode(vec, SLOTS, BUDGET)
    assert all(not s.is_active() for s in cfg.sensors)


def test_fixed_mount_order_uses_row_index_for_slot():
    """Row i must map to mounting_slots[i] regardless of slot float in the vector."""
    six = ["a", "b", "c", "d", "e", "f"]
    b6 = {
        "lidar": {"max_count": 2},
        "camera": {"max_count": 2},
        "radar": {"max_count": 2},
    }
    vec = np.zeros(6 * FLOATS_PER_SENSOR, dtype=np.float64)
    for i in range(6):
        base = i * FLOATS_PER_SENSOR
        vec[base + 0] = 1.0  # lidar
        vec[base + 1] = 1.0
        vec[base + 2] = 0.0  # would be slot "a" in non-fixed mode, not six[i] if i > 0
    cfg = decode(vec, six, b6, fixed_mount_order=True)
    for i, s in enumerate(cfg.sensors):
        assert s.slot == six[i]


def test_decode_budget_enforcement():
    budget = {"lidar": {"max_count": 2}}
    n = 3
    vec = np.zeros(n * FLOATS_PER_SENSOR)
    for i in range(n):
        base = i * FLOATS_PER_SENSOR
        vec[base + 0] = 1.0
        vec[base + 1] = 1.0
    cfg = decode(vec, SLOTS, budget)
    active_lidars = [s for s in cfg.sensors if s.sensor_type == "lidar"]
    assert len(active_lidars) <= 2


def test_decode_slots_in_valid_set():
    rng = np.random.default_rng(0)
    for _ in range(20):
        vec = rng.uniform(-1.0, 4.0, size=config_vector_size(BUDGET))
        cfg = decode(vec, SLOTS, BUDGET)
        for s in cfg.sensors:
            assert s.slot in SLOTS


def test_decode_offsets_clamped():
    rng = np.random.default_rng(1)
    for _ in range(20):
        vec = rng.uniform(-5.0, 5.0, size=config_vector_size(BUDGET))
        cfg = decode(vec, SLOTS, BUDGET)
        for s in cfg.sensors:
            assert -0.5 <= s.x_offset <= 0.5
            assert -0.5 <= s.y_offset <= 0.5
            assert  0.0 <= s.z_offset <= 0.5


def test_decode_range_fraction_clamped():
    rng = np.random.default_rng(2)
    vec = rng.uniform(-5.0, 5.0, size=config_vector_size(BUDGET))
    cfg = decode(vec, SLOTS, BUDGET)
    for s in cfg.sensors:
        assert 0.1 <= s.range_fraction <= 1.0
        assert 0.2 <= s.hfov_fraction  <= 1.0


def test_encode_decode_roundtrip_active_sensors():
    original = SensorConfig(sensors=[
        SingleSensorConfig("lidar",  "top",   z_offset=0.3,  yaw_deg=0.0,  range_fraction=1.0, hfov_fraction=1.0),
        SingleSensorConfig("camera", "front", z_offset=0.2,  yaw_deg=10.0, range_fraction=0.8, hfov_fraction=0.9),
    ])
    vec = encode(original, SLOTS)
    recovered = decode(vec, SLOTS, BUDGET_SMALL)
    orig_active  = original.active_sensors()
    recov_active = recovered.active_sensors()
    assert len(recov_active) == len(orig_active)
    for orig_s, rec_s in zip(orig_active, recov_active):
        assert orig_s.sensor_type == rec_s.sensor_type
        assert orig_s.slot == rec_s.slot
        assert abs(orig_s.z_offset       - rec_s.z_offset)       < 1e-3
        assert abs(orig_s.range_fraction - rec_s.range_fraction)  < 1e-3
        assert abs(orig_s.hfov_fraction  - rec_s.hfov_fraction)   < 1e-3


def test_encode_decode_roundtrip_disabled():
    original = SensorConfig(sensors=[
        SingleSensorConfig("disabled", "front"),
        SingleSensorConfig("disabled", "rear"),
    ])
    vec = encode(original, SLOTS)
    recovered = decode(vec, SLOTS, BUDGET)
    assert not any(s.is_active() for s in recovered.sensors)


def test_roundtrip_yaw_approx():
    for yaw in [-180.0, -90.0, 0.0, 45.0, 90.0, 179.0]:
        original = SensorConfig(sensors=[SingleSensorConfig("lidar", "top", yaw_deg=yaw)])
        vec = encode(original, SLOTS)
        recovered = decode(vec, SLOTS, {"lidar": {"max_count": 1}})
        active = recovered.active_sensors()
        assert len(active) == 1
        assert abs(active[0].yaw_deg - yaw) < 1.0


def test_count_by_type():
    cfg = make_two_sensor_config()
    counts = cfg.count_by_type()
    assert counts.get("lidar",  0) == 1
    assert counts.get("camera", 0) == 1
    assert counts.get("radar",  0) == 1
    assert "disabled" not in counts


def test_total_cost():
    cfg = SensorConfig(sensors=[
        SingleSensorConfig("lidar",  "top"),
        SingleSensorConfig("camera", "front"),
    ])
    sensor_models = {"lidar": {"cost_usd": 4000}, "camera": {"cost_usd": 200}}
    assert cfg.total_cost(sensor_models) == 4200.0


def test_active_sensors_excludes_disabled():
    cfg = SensorConfig(sensors=[
        SingleSensorConfig("lidar",    "top"),
        SingleSensorConfig("disabled", "front"),
        SingleSensorConfig("camera",   "rear"),
    ])
    active = cfg.active_sensors()
    assert len(active) == 2
    assert all(s.is_active() for s in active)


def test_fixed_sensor_geometry_trims_vector_length():
    b2 = {"lidar": {"max_count": 1}, "camera": {"max_count": 1}}
    assert floats_per_sensor(True) == 2
    assert config_vector_size(b2, fixed_sensor_geometry=True) == 4
    v = make_initial_vector(b2, ["a", "b"], fixed_sensor_geometry=True)
    assert v.shape == (4,)
    assert (v == 0).all()


def test_decode_fixed_sensor_geometry_allocation_only():
    six = ["m0", "m1", "m2", "m3", "m4", "m5"]
    b6 = {
        "lidar":  {"max_count": 2},
        "camera": {"max_count": 2},
        "radar":  {"max_count": 2},
    }
    vec = np.zeros(12, dtype=np.float64)
    vec[0:2] = (2.0, 1.0)  # camera, active
    # remaining slots: disabled
    dpose = {
        "all": {
            "yaw_deg": 0.0,
            "pitch_deg": 0.0,
            "z_offset": 0.2,
        },
        "camera": {
            "pitch_deg": 0.0,
        },
    }
    cfg = decode(
        vec, six, b6,
        fixed_sensor_geometry=True,
        fixed_mount_order=True,
        default_sensor_pose=dpose,
    )
    assert cfg.sensors[0].sensor_type == "camera"
    assert cfg.sensors[0].slot == "m0"
    assert cfg.sensors[0].yaw_deg == 0.0
    assert cfg.sensors[0].pitch_deg == 0.0
    assert cfg.sensors[0].z_offset == 0.2
    for s in cfg.sensors[1:]:
        assert s.sensor_type == "disabled"

    w = merge_default_sensor_pose("camera", "m0", dpose)
    assert w["yaw_deg"] == 0.0 and w["pitch_deg"] == 0.0