"""Tests for user-entered sensor, quantity, and hardware specs."""

from __future__ import annotations

import pytest

from sensor_opt.config.specs import (
    prepare_experiment_config,
    quantity_values,
    validate_experiment_specs,
)
from sensor_opt.config.specs import normalize_sensor_budget_inplace


def _base_cfg(mode: str = "mock_isaac") -> dict:
    return {
        "sensor_budget": {
            # Inventory-style caps: up to 2 lidar, 3 camera (may use 0..N of each; not forced to 5 total).
            "lidar": {"usermax": 2, "max_count": 2},
            "camera": {"usermax": 3, "max_count": 3},
        },
        "sensor_models": {
            "lidar": {
                "cost_usd": 4000,
                "range_m": 100.0,
                "compute_tops": 8.0,
                "memory_gb": 1.2,
                "latency_ms": 20.0,
            },
            "camera": {
                "cost_usd": 200,
                "range_m": 10.0,
                "compute_tops": 2.0,
                "memory_gb": 0.4,
                "latency_ms": 8.0,
            },
        },
        "inner_loop": {"mode": mode},
    }


def test_prepare_copies_usermax_to_max_count() -> None:
    cfg: dict = {
        "sensor_budget": {
            "lidar": {"usermax": 6},
            "camera": {"usermax": 3},
        },
        "sensor_models": {
            "lidar": {"cost_usd": 1},
            "camera": {"cost_usd": 1},
        },
        "inner_loop": {"mode": "mock_isaac"},
    }
    out = prepare_experiment_config(cfg)
    assert out["sensor_budget"]["lidar"]["max_count"] == 6
    assert out["sensor_budget"]["camera"]["max_count"] == 3
    # Original untouched
    assert "max_count" not in cfg["sensor_budget"]["lidar"]


def test_quantity_range_is_zero_through_max() -> None:
    cfg = _base_cfg()
    assert quantity_values(cfg["sensor_budget"], "lidar") == [0, 1, 2]
    assert quantity_values(cfg["sensor_budget"], "camera") == [0, 1, 2, 3]


def test_min_count_limits_low_end() -> None:
    cfg = _base_cfg()
    cfg["sensor_budget"]["camera"] = {"usermax": 3, "max_count": 3, "min_count": 1}
    normalize_sensor_budget_inplace(cfg)
    assert quantity_values(cfg["sensor_budget"], "camera") == [1, 2, 3]


def test_mock_isaac_does_not_require_hardware_specs() -> None:
    validate_experiment_specs(_base_cfg(mode="mock_isaac"))


def test_isaac_sim_requires_machine_hardware_specs() -> None:
    with pytest.raises(ValueError, match="hardware specs are required"):
        validate_experiment_specs(_base_cfg(mode="isaac_sim"))


def test_isaac_sim_accepts_numerical_machine_hardware_specs() -> None:
    cfg = _base_cfg(mode="isaac_sim")
    cfg["hardware"] = {
        "name": "apple_m_like",
        "gpu_cores": 10,
        "unified_memory_gb": 24,
        "memory_bandwidth_gbps": 100,
    }
    validate_experiment_specs(cfg)


def test_usermax_and_max_count_mismatch_rejected() -> None:
    cfg = _base_cfg()
    cfg["sensor_budget"]["lidar"] = {"usermax": 3, "max_count": 2}
    with pytest.raises(ValueError, match="usermax and max_count must be equal"):
        prepare_experiment_config(cfg)
