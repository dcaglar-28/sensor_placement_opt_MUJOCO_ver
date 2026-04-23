import numpy as np

from sensor_opt.inner_loop.isaaclab_ground_robot import min_forward_range_from_obs, min_range_any_from_obs


def test_min_forward_range_from_lidar_points():
    # Forward +X sector: point straight ahead
    pts = np.array(
        [
            [5.0, 0.0, 0.0],  # in-FOV
            [5.0, 10.0, 0.0],  # out of forward cone, should be ignored
        ],
        dtype=np.float64,
    )
    obs = {"policy": {"lidar": pts}}
    d = min_forward_range_from_obs(obs, env_idx=0, sensor_models={})
    assert d is not None
    assert abs(float(d) - 5.0) < 1e-6


def test_min_range_any_prefers_min_norm():
    pts = np.array(
        [
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    obs = {"lidar": pts}
    d = min_range_any_from_obs(obs, env_idx=0)
    assert d is not None
    assert abs(float(d) - 1.0) < 1e-6
