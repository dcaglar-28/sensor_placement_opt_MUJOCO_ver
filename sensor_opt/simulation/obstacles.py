"""Obstacle layout shared by all candidates within a CMA-ES generation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

Position = Tuple[float, float, float]


def get_generation_seed(generation_number: int, base_seed: int = 42) -> int:
    """seed = base_seed * 1000 + generation_number (1-based CMA gen recommended)."""
    return int(base_seed) * 1000 + int(generation_number)


def generate_obstacles(
    n_obstacles: int, path_length_m: float, rng_seed: int
) -> List[Position]:
    """
    Obstacle centers on the ground plane (Z = 0) for the kinematic sim.
    X: [2.0, path_length_m - 1.0], Y: [-1.5, 1.5], Z: 0.0
    """
    n = int(max(0, n_obstacles))
    lo_x = 2.0
    hi_x = float(path_length_m) - 1.0
    if hi_x < lo_x:
        hi_x = lo_x
    rng = np.random.default_rng(int(rng_seed))
    out: List[Position] = []
    for _ in range(n):
        x = float(rng.uniform(lo_x, hi_x))
        y = float(rng.uniform(-1.5, 1.5))
        out.append((x, y, 0.0))
    return out
