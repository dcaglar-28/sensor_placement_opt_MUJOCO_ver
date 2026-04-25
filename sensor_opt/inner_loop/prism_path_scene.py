"""
Prism + path layout for the rectangular-prism sensor-placement testbed.

Body frame: +X = forward (front face), +Y = left, +Z = up. The prism is axis-aligned
with half-extents (sx, sy, sz) in meters (sx = half length along +X, etc.).

The six mount points (named consistently with `configs/obstacle_isaaclab.yaml`):

- Two on the *front* face (+X) near the face center, split **left / right** along Y
  (avoids a duplicate single centroid).
- Two on the **front top** edge (intersection of front face and top face), along that
  edge split into **left and right** segments.
- One at the midpoint of the **front-left** vertical edge, one at the **front-right**
  vertical edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

# Stable keys: order matches `mounting_slots` in obstacle_isaaclab (fixed_mount_order)
PRISM_MOUNT_NAMES: List[str] = [
    "prism_front_face_l",  # front face, +Y side of center
    "prism_front_face_r",  # front face, -Y side of center
    "prism_top_edge_l",  # top front edge, +Y half
    "prism_top_edge_r",  # top front edge, -Y half
    "prism_left_edge",  # front-left vertical edge midpoint
    "prism_right_edge",  # front-right vertical edge midpoint
]


@dataclass(frozen=True)
class ObstacleVolume:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_surface: float


def default_corridor_volume() -> ObstacleVolume:
    return ObstacleVolume(2.0, 8.0, -2.0, 2.0, 0.0)  # z set per obstacle size in bridge


def default_prism_path_volume() -> ObstacleVolume:
    """A band along a forward path similar to a moving prism; wider X span than the old corridor band."""
    return ObstacleVolume(0.5, 9.5, -1.5, 1.5, 0.0)


def prism_sensor_local_translations_m(
    sx: float, sy: float, sz: float
) -> dict[str, tuple[float, float, float]]:
    """
    Return local (x,y,z) translation for each mount's Xform relative to the prism
    body (centered) frame.
    """
    sx, sy, sz = float(sx), float(sy), float(sz)
    y_face = 0.35 * sy
    y_top = 0.4 * sy
    return {
        "prism_front_face_l": (sx, y_face, 0.0),
        "prism_front_face_r": (sx, -y_face, 0.0),
        "prism_top_edge_l": (sx, y_top, sz),
        "prism_top_edge_r": (sx, -y_top, sz),
        "prism_left_edge": (sx, sy, 0.0),
        "prism_right_edge": (sx, -sy, 0.0),
    }


def prism_body_world_x_along_path(
    t_s: float,
    *,
    t_episode_s: float,
    x0: float = 0.5,
    x1: float = 8.0,
) -> float:
    """One-way motion along +X for duration of an episode; holds at the end."""
    t_ep = max(float(t_episode_s), 1e-6)
    a = min(1.0, max(0.0, float(t_s) / t_ep))
    return x0 + a * (float(x1) - float(x0))


def sample_obstacle_positions(
    rng: np.random.Generator,
    n: int,
    size_m: float,
    vol: ObstacleVolume,
) -> list[tuple[float, float, float]]:
    n = int(max(1, n))
    s2 = 0.5 * float(size_m)
    out: list[tuple[float, float, float]] = []
    for _ in range(n):
        x = float(rng.uniform(vol.x_min, vol.x_max))
        y = float(rng.uniform(vol.y_min, vol.y_max))
        z = s2 if vol.z_surface == 0.0 else float(vol.z_surface) + s2
        out.append((x, y, z))
    return out


def build_mount_prim_paths(
    base_root: str, env_idx: int = 0, names: Optional[List[str]] = None
) -> dict[str, str]:
    """
    USD Xform paths for bridge-spawned prototype prims:
      {base}/env_{i}/PrismBody/mounts/{name}
    `base` should be like `/World/bridge_prism_path` (no trailing env segment).
    """
    names = names or PRISM_MOUNT_NAMES
    b = str(base_root).rstrip("/")
    pre = f"{b}/env_{int(env_idx)}/PrismBody/mounts"
    return {n: f"{pre}/{n}" for n in names}
