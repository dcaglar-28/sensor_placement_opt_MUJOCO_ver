"""
Per-type minima for `sensor_budget.min_count` when `inner_loop.mode: mujoco`.

* **min_count** — What the MuJoCo inner loop needs to run in a *well-posed* way
  (at least for validation / search lower bounds). Defined here, not as “user
  resources”. Override per-type via `inner_loop.mujoco.sim_min_count` in YAML.

* **usermax / max_count** — User’s inventory (how many units they can place); set
  in `sensor_budget` in YAML. `prepare_experiment_config` normalizes `usermax`
  → `max_count` when only `usermax` is set.
"""

from __future__ import annotations

from typing import Dict

# All types in sensor_budget; missing keys default to 0.
DEFAULT_SIM_MIN_COUNT: Dict[str, int] = {
    "lidar": 0,
    "camera": 0,
    "radar": 0,
}


def mujoco_sim_min_count() -> Dict[str, int]:
    """
    Return default per-type minimum active sensors required by the current MuJoCo
    scenarios (prism + planar). Increase a value when a task truly requires e.g.
    at least one camera to execute rollouts.
    """
    return dict(DEFAULT_SIM_MIN_COUNT)
