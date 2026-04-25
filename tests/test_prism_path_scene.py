import numpy as np
import pytest

from sensor_opt.inner_loop.prism_path_scene import (
    PRISM_MOUNT_NAMES,
    build_mount_prim_paths,
    default_prism_path_volume,
    prism_body_world_x_along_path,
    prism_sensor_local_translations_m,
    sample_obstacle_positions,
)


def test_six_mount_names():
    assert len(PRISM_MOUNT_NAMES) == 6
    assert len(set(PRISM_MOUNT_NAMES)) == 6


def test_prism_local_offsets_reasonable():
    t = prism_sensor_local_translations_m(0.5, 0.3, 0.2)
    for k in PRISM_MOUNT_NAMES:
        assert k in t
    assert t["prism_front_face_l"][0] > 0.0
    assert t["prism_top_edge_l"][-1] == pytest.approx(0.2)


def test_path_monotone():
    v = [prism_body_world_x_along_path(i * 0.1, t_episode_s=1.0) for i in range(11)]
    assert v[0] <= v[-1]


def test_build_mount_prim_paths():
    m = build_mount_prim_paths("/World/bridge_prism_path", env_idx=0)
    assert m["prism_left_edge"] == (
        "/World/bridge_prism_path/env_0/PrismBody/mounts/prism_left_edge"
    )


def test_sample_obstacles_in_volume():
    vol = default_prism_path_volume()
    rng = np.random.default_rng(0)
    pts = sample_obstacle_positions(rng, 10, 0.5, vol)
    for x, y, z in pts:
        assert vol.x_min <= x <= vol.x_max
        assert vol.y_min <= y <= vol.y_max
        assert z > 0.0
