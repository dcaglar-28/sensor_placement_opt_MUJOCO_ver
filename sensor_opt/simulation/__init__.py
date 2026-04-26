"""MuJoCo kinematic vehicle + fixed-slot sensor evaluation."""

from sensor_opt.simulation.obstacles import generate_obstacles, get_generation_seed
from sensor_opt.simulation.mujoco_runner import run_episode
from sensor_opt.simulation.sensor_specs import get_sensor_specs, merge_sensor_spec_overrides

__all__ = [
    "generate_obstacles",
    "get_generation_seed",
    "run_episode",
    "get_sensor_specs",
    "merge_sensor_spec_overrides",
]
