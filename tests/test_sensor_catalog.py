from __future__ import annotations

import pytest

from sensor_opt.config.catalog import apply_sensor_catalog


def test_apply_sensor_catalog_noop_when_sensor_models_present():
    cfg = {"sensor_models": {"lidar": {"cost_usd": 1.0}}}
    out = apply_sensor_catalog(cfg)
    assert out is cfg
    assert out["sensor_models"]["lidar"]["cost_usd"] == 1.0


def test_apply_sensor_catalog_synthesizes_sensor_models():
    cfg = {
        "sensor_catalog": {
            "vlp16": {"sensor_type": "lidar", "cost_usd": 4000, "compute_tops": 8.0},
            "d435i": {"sensor_type": "camera", "cost_usd": 200, "compute_tops": 2.0},
        },
        "sensor_choices": {"lidar": "vlp16", "camera": "d435i"},
    }
    out = apply_sensor_catalog(cfg)
    assert out is not cfg
    assert out["sensor_models"]["lidar"]["cost_usd"] == 4000
    assert out["sensor_models"]["camera"]["compute_tops"] == 2.0
    assert "sensor_type" not in out["sensor_models"]["lidar"]


def test_apply_sensor_catalog_raises_on_unknown_model_id():
    cfg = {
        "sensor_catalog": {"vlp16": {"sensor_type": "lidar", "cost_usd": 4000}},
        "sensor_choices": {"lidar": "nope"},
    }
    with pytest.raises(KeyError):
        apply_sensor_catalog(cfg)

