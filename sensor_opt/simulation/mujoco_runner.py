"""
Kinematic single-path episode: mj_kinematics only (no mj_step). Geometric ray–sphere vs obstacles.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sensor_opt.simulation.mjcf import SLOT_NAMES

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]

_R_SPHERE = 0.25

# z center for generated obstacles (ground-level centers from YAML use z=0 -> lift to clear plane)
def _oc_z(raw_z: float) -> float:
    if abs(raw_z) < 1e-6:
        return 0.2
    return float(raw_z)


def _ray_sphere(p0: np.ndarray, u: np.ndarray, c: np.ndarray, r: float) -> float | None:
    """Return forward distance along +u to first intersection with sphere, or None."""
    oc = p0 - c
    b = float(np.dot(u, oc))
    det = b * b - (float(np.dot(oc, oc)) - r * r)
    if det < 0.0:
        return None
    sd = math.sqrt(det)
    t0 = -b - sd
    t1 = -b + sd
    for t in (t0, t1):
        if t >= 1e-7:
            return t
    return None


def _cone_rays_local(n_rays: int, half_fov_rad: float, rng: np.random.Generator) -> List[np.ndarray]:
    """Unit directions in vehicle/body frame with mean along +X (forward)."""
    dirs: List[np.ndarray] = []
    n_r = max(1, n_rays)
    for _ in range(n_r):
        if half_fov_rad < 1e-6:
            dirs = [np.array([1.0, 0.0, 0.0], dtype=np.float64)]
            break
        uu = float(rng.uniform(0.0, 1.0))
        cos_t = 1.0 - uu * (1.0 - math.cos(half_fov_rad))
        sin_t = math.sqrt(max(0.0, 1.0 - cos_t * cos_t))
        phi = float(rng.uniform(0.0, 2.0 * math.pi))
        dx = cos_t
        dy = sin_t * math.sin(phi)
        dz = sin_t * math.cos(phi)
        d = np.array([dx, dy, dz], dtype=np.float64)
        d = d / (float(np.linalg.norm(d)) + 1e-9)
        dirs.append(d)
    return dirs


def run_episode(
    model: "mujoco.MjModel",
    data: "mujoco.MjData",
    config: Dict[str, str],
    obstacle_positions: List[Tuple[float, float, float]],
    sim_cfg: Dict[str, Any],
    sensor_specs: Dict[str, Dict[str, Any]],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Args:
        config: {slot_name: sensor type string}
        obstacle_positions: one tuple per in-use obstacle (x,y,z) world; unused slots in XML parked separately
    """
    if mujoco is None:  # pragma: no cover
        raise ImportError("mujoco is required for run_episode")

    path_len = float(sim_cfg.get("path_length_m", 20.0))
    v = float(sim_cfg.get("vehicle_speed_mps", 2.0))
    dt = float(sim_cfg.get("timestep_s", 0.02))
    n_obstacles = int(sim_cfg.get("n_obstacles", 10))

    mujoco.mj_resetData(model, data)
    v_j = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "vehicle_tx"))
    v_q = int(model.jnt_qposadr[v_j])

    site_ids = [int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)) for s in SLOT_NAMES]
    mocap_ids: list[int] = []
    for i in range(n_obstacles):
        bid = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"obst_{i}"))
        mocap_ids.append(int(model.body_mocapid[bid]))

    for i in range(n_obstacles):
        mid = mocap_ids[i]
        if i < len(obstacle_positions):
            ox, oy, oz = obstacle_positions[i]
            data.mocap_pos[mid, 0] = float(ox)
            data.mocap_pos[mid, 1] = float(oy)
            data.mocap_pos[mid, 2] = _oc_z(oz)
        else:
            data.mocap_pos[mid, :] = [1000.0, 0.0, 0.2]

    n_use = min(len(obstacle_positions), n_obstacles)
    ocenters: list[np.ndarray] = []
    for i in range(n_use):
        ox, oy, oz = obstacle_positions[i]
        ocenters.append(np.array([float(ox), float(oy), _oc_z(oz)], dtype=np.float64))

    duration = path_len / max(v, 1e-6)
    n_steps = int(math.ceil(duration / max(dt, 1e-9))) + 1

    first_t: list[Optional[float]] = [None] * n_use
    first_d: list[Optional[float]] = [None] * n_use
    slot_first_hits: Dict[str, int] = {s: 0 for s in SLOT_NAMES}
    n_cov = 0

    veh_b = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "vehicle"))

    for step in range(n_steps):
        t_now = float(step) * dt
        data.qpos[v_q] = v * t_now
        if hasattr(mujoco, "mj_kinematics"):
            mujoco.mj_kinematics(model, data)
        else:  # pragma: no cover
            mujoco.mj_forward(model, data)

        # Body rotation: vehicle only slides; boresight +X in world
        xm = np.asarray(data.xmat[veh_b], dtype=np.float64).ravel()
        Rm = xm.reshape(3, 3) if xm.size == 9 else np.eye(3, dtype=np.float64)
        step_coverage = False

        for sidx, slot in enumerate(SLOT_NAMES):
            st = str(config.get(slot, "disabled"))
            if st == "disabled":
                continue
            spec = sensor_specs.get(st) or {}
            fov = float(spec.get("fov_deg", 60.0))
            r_max = float(spec.get("max_range_m", 10.0))
            n_rays = int(spec.get("n_rays", 8))
            half_f = math.radians(0.5 * min(fov, 175.0))

            p0 = np.array(data.site_xpos[site_ids[sidx]], dtype=np.float64)
            for d_l in _cone_rays_local(n_rays, half_f, rng):
                d_w = Rm @ d_l
                d_w = d_w / (float(np.linalg.norm(d_w)) + 1e-9)
                best: tuple[int, float] | None = None
                for oi, oc in enumerate(ocenters):
                    t_hit = _ray_sphere(p0, d_w, oc, _R_SPHERE)
                    if t_hit is None or t_hit > r_max + 1e-6:
                        continue
                    if best is None or t_hit < best[1]:
                        best = (oi, t_hit)
                if best is not None:
                    oi, t_hit = best
                    step_coverage = True
                    if first_t[oi] is None:
                        first_t[oi] = t_now
                        first_d[oi] = t_hit
                        slot_first_hits[slot] = int(slot_first_hits.get(slot, 0)) + 1
        if step_coverage:
            n_cov += 1

    n_det = sum(1 for f in first_t if f is not None)
    dists = [d for d in first_d if d is not None]
    mean_d = float(np.mean(dists)) if dists else 0.0
    cov_frac = float(n_cov) / float(max(1, n_steps))

    return {
        "first_detection_times": [None if f is None else float(f) for f in first_t],
        "coverage_fraction": cov_frac,
        "n_detected": int(n_det),
        "mean_detection_distance": mean_d,
        "n_obstacles": int(n_use),
        "per_slot_first_hits": dict(slot_first_hits),
        "episode_duration_s": float(duration),
    }
