"""
Tests for prim-path resolution helpers in `scripts/isaaclab_sensor_bridge.py` (no Isaac import).
"""

import importlib.util
import os
from pathlib import Path

import pytest

_BRIDGE = Path(__file__).resolve().parents[1] / "scripts" / "isaaclab_sensor_bridge.py"


def _load_bridge_mod():
    spec = importlib.util.spec_from_file_location("isaaclab_sensor_bridge", str(_BRIDGE))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def test_resolve_prims_mount_paths():
    b = _load_bridge_mod()
    m = {
        "mount_prim_paths": {"top": "/World/envs/env_{env_idx}/L/top"},
    }
    p = b._resolve_isaac_prim_path(m, "lidar", "top", 0, 1)
    assert p == "/World/envs/env_1/L/top"


def test_resolve_prims_list_and_env():
    b = _load_bridge_mod()
    m = {
        "prim_paths": ["/A/{env_idx}/0", "/A/{env_idx}/1"],
    }
    assert b._resolve_isaac_prim_path(m, "lidar", "front", 0, 2) == "/A/2/0"
    assert b._resolve_isaac_prim_path(m, "lidar", "front", 1, 2) == "/A/2/1"
    m2 = {"prim_path": "/Only"}
    assert b._resolve_isaac_prim_path(m2, "lidar", "top", 0, 0) == "/Only"
    os.environ["ISAAC_CAMERA_PRIM"] = "/FromEnv{env_idx}"
    try:
        assert b._resolve_isaac_prim_path({}, "camera", "front", 0, 3) == "/FromEnv3"
    finally:
        del os.environ["ISAAC_CAMERA_PRIM"]


def test_euler_to_quat_finite():
    b = _load_bridge_mod()
    w, x, y, z = b._euler_rpy_deg_to_quat_wxyz(0.0, 0.0, 0.0)
    assert pytest.approx(w, rel=0, abs=1e-9) == 1.0
    assert abs(x) + abs(y) + abs(z) < 1e-6
