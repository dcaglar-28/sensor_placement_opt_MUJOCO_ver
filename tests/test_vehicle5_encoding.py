"""Five-slot [0,1] genome: no per-slot offsets, bucketed types only."""

import numpy as np

from sensor_opt.encoding.config import _gene_to_type_01, config_vector_size, decode
from sensor_opt.simulation.obstacles import generate_obstacles, get_generation_seed


def test_genome_5d_decode_buckets():
    budget = {
        "lidar": {"max_count": 5, "min_count": 0},
        "camera": {"max_count": 5, "min_count": 0},
        "radar": {"max_count": 5, "min_count": 0},
        "disabled": {"max_count": 5, "min_count": 0},
    }
    slots = [
        "top_front_edge_l",
        "top_front_edge_r",
        "front_face_center",
        "side_front_edge_l",
        "side_front_edge_r",
    ]
    vec = np.array([0.1, 0.3, 0.55, 0.8, 0.99])
    c = decode(vec, slots, budget, vehicle_5slot=True)
    types = [s.sensor_type for s in c.sensors]
    assert types[0] == "disabled"
    assert types[1] == "camera"
    assert types[2] == "radar"
    assert types[3] == "lidar"
    assert types[4] == "lidar"


def test_config_vector_size_vehicle():
    b = {
        "lidar": {"max_count": 5, "min_count": 0},
        "camera": {"max_count": 5, "min_count": 0},
        "radar": {"max_count": 5, "min_count": 0},
        "disabled": {"max_count": 5, "min_count": 0},
    }
    assert config_vector_size(b, vehicle_5slot=True) == 5


def test_max_sensor_count_merges_to_two_active():
    budget = {
        "lidar": {"max_count": 5, "min_count": 0},
        "camera": {"max_count": 5, "min_count": 0},
        "radar": {"max_count": 5, "min_count": 0},
        "disabled": {"max_count": 5, "min_count": 0},
    }
    slots = [
        "top_front_edge_l",
        "top_front_edge_r",
        "front_face_center",
        "side_front_edge_l",
        "side_front_edge_r",
    ]
    vec = np.array([0.4, 0.4, 0.4, 0.4, 0.4])
    c = decode(vec, slots, budget, vehicle_5slot=True, max_sensor_count=2)
    assert len([s for s in c.sensors if s.is_active()]) == 2


def test_obstacle_generation_seed_per_generation():
    a = generate_obstacles(10, 20.0, get_generation_seed(5, 42))
    b = generate_obstacles(10, 20.0, get_generation_seed(5, 42))
    c = generate_obstacles(10, 20.0, get_generation_seed(6, 42))
    assert a == b
    assert a != c


def test_gene_clip():
    assert _gene_to_type_01(0.24) == "disabled"
    assert _gene_to_type_01(0.25) == "camera"
    assert _gene_to_type_01(0.5) == "radar"
    assert _gene_to_type_01(0.99) == "lidar"
