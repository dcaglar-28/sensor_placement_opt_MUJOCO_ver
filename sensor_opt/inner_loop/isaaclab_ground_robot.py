"""
Ground-robot perception helpers for Isaac Lab rollouts (best-effort, task-agnostic).

This module is intentionally free of `isaaclab` imports so it can be unit-tested in CI.

The core output used by the bridge is a scalar *blind spot* proxy in [0, 1]:
  blind ≈ 1 - mean(union(coverage_lidar, coverage_depth) over horizontal bins)

It is a geometric heuristic, not a ground-truth "occlusion" metric, but it matches the
project's need for a single `blind_spot_fraction` scalar for optimization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class GroundCoverageConfig:
    angular_bins: int = 72  # 5° bins around the full 360° horizon
    min_lidar_points_per_bin: int = 1
    depth_valid_min: float = 1e-3
    # treat very large depth as invalid / missing (sensor-dependent)
    depth_valid_max: float = 50.0


def _to_numpy(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        import torch  # type: ignore

        if torch.is_tensor(x):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def _row(x: Any, env_idx: int) -> Any:
    """Index the first ("batch/env") dimension for common Isaac Lab tensor shapes."""
    if x is None:
        return None
    if isinstance(x, dict):
        return x
    try:
        import torch  # type: ignore

        if torch.is_tensor(x):
            return x[env_idx]
    except Exception:
        pass
    try:
        arr = np.asarray(x)
        return arr[env_idx]
    except Exception:
        return x


def _unwrap_sensor_obs_dict(o: Any) -> Any:
    """
    Unwrap one level: ``{"policy": {"lidar": ...}}`` → ``{"lidar": ...}`` so
    per-modality keys are not hidden behind a common RL/bridge wrapper.
    """
    if not isinstance(o, dict) or not o:
        return o
    for k in ("policy", "obs", "observations", "sensors", "perception", "sensors_state"):
        if k in o and isinstance(o[k], dict) and o[k]:
            inner = o[k]
            names = " ".join(str(ik).lower() for ik in inner.keys())
            if any(
                s in names
                for s in (
                    "lidar",
                    "point",
                    "pc",
                    "cloud",
                    "depth",
                    "range",
                    "returns",
                    "camera",
                    "rgb",
                )
            ):
                return inner
    return o


def _iter_tensor_leaves(obj: Any) -> Iterable[np.ndarray]:
    """
    Yields leaf arrays/tensors (converted to numpy) from a nested structure.
    This is a pragmatic way to find depth / lidar without hard-coding exact keys.
    """
    if obj is None:
        return
    if isinstance(obj, (str, bytes)):
        return

    # dict-like
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_tensor_leaves(v)
        return

    # tuple/list
    if isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_tensor_leaves(v)
        return

    a = _to_numpy(obj)
    if a is None:
        return
    if isinstance(a, np.ndarray) and a.dtype != object and a.size > 0:
        yield a


def _lidar_coverage(pts_xyz: np.ndarray, cfg: GroundCoverageConfig) -> float:
    """
    pts_xyz: (N,3) in world or robot frame; we use x-y plane heading only.
    Returns coverage in [0,1] as mean over angular bins: 1 if bin has enough points.
    """
    p = np.asarray(pts_xyz, dtype=np.float64)
    if p.size == 0:
        return 0.0
    if p.ndim != 2 or p.shape[1] < 3:
        # try reshape if looks like (3,N)
        p = p.reshape(-1, p.shape[-1])
        if p.shape[1] < 3:
            return 0.0
    xy = p[:, :2]
    norms = np.linalg.norm(xy, axis=1)
    if np.max(norms) < 1e-6:
        return 0.0
    ang = np.arctan2(xy[:, 1], xy[:, 0])  # [-pi, pi]
    bin_idx = np.floor((ang + math.pi) / (2 * math.pi) * cfg.angular_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, cfg.angular_bins - 1)
    counts = np.bincount(bin_idx, minlength=cfg.angular_bins)
    covered = counts >= cfg.min_lidar_points_per_bin
    return float(np.mean(covered)) if covered.size else 0.0


def _depth_coverage(depth: np.ndarray, hfov_deg: float, cfg: GroundCoverageConfig) -> float:
    """
    depth: 2D array (H, W) or (1,H,W) etc.
    Heuristic: map each column to an azimuth bin within camera HFOV, mark covered if
    there exists a valid depth pixel in that column.
    """
    d = np.asarray(depth)
    if d.size == 0:
        return 0.0
    # squeeze to 2D
    while d.ndim > 2 and d.shape[0] == 1:
        d = d[0]
    if d.ndim != 2:
        # try last-two dims
        if d.ndim >= 2:
            d = d.reshape(-1, d.shape[-2], d.shape[-1])[-1]
    if d.ndim != 2:
        return 0.0

    h, w = d.shape
    # middle band to reduce edge artifacts
    h0, h1 = int(0.25 * h), int(0.75 * h)
    band = d[h0:h1, :]

    half = float(hfov_deg) * 0.5
    cov = np.zeros(cfg.angular_bins, dtype=bool)
    for col in range(w):
        # angle linearly across image columns: [-HFOV/2, +HFOV/2]
        t = (col + 0.5) / max(w, 1)
        ang_deg = (t - 0.5) * float(hfov_deg)
        if abs(ang_deg) > half + 1e-6:
            continue
        # map to global 360 bin space centered around robot forward (+X): shift by +90°? :
        # Without extrinsics, this is a *relative* coverage metric, still useful for ranking.
        ang = math.radians(ang_deg)
        bin_i = int(math.floor((ang + math.pi) / (2 * math.pi) * cfg.angular_bins))
        bin_i = int(np.clip(bin_i, 0, cfg.angular_bins - 1))
        col_vals = band[:, col]
        valid = (col_vals > cfg.depth_valid_min) & (col_vals < cfg.depth_valid_max)
        if bool(np.any(valid)):
            cov[bin_i] = True
    return float(np.mean(cov)) if cov.size else 0.0


def estimate_blind_spot_fraction_from_obs(
    obs: Any,
    *,
    env_idx: int,
    sensor_models: dict,
    cfg: GroundCoverageConfig | None = None,
) -> Optional[float]:
    """
    Return blind_spot_fraction in [0,1] if we find plausible depth and/or 3D points; else None.
    """
    cfg = cfg or GroundCoverageConfig()
    if obs is None:
        return None

    o = _row(obs, env_idx)

    # Collect candidate arrays
    best_depth: Optional[np.ndarray] = None
    best_lidar: Optional[np.ndarray] = None

    # Prefer common names if present
    if isinstance(o, dict):
        for k, v in o.items():
            kl = str(k).lower()
            a = _to_numpy(v)
            if a is None:
                continue
            if any(s in kl for s in ("depth", "distance", "z_buf")) and best_depth is None:
                best_depth = a
            if any(s in kl for s in ("lidar", "range", "point", "pc", "cloud")) and best_lidar is None:
                # might still be wrong, we'll validate shapes later
                best_lidar = a

    # If not found, scan leaves for plausible shapes
    if best_depth is None or best_lidar is None:
        for a in _iter_tensor_leaves(o):
            a = np.asarray(a)
            if a.size == 0:
                continue
            # depth: 2D
            if best_depth is None and a.ndim >= 2 and a.shape[-1] > 4 and a.ndim == 2:
                best_depth = a
                continue
            # points: (N,3) or (3,N)
            if best_lidar is None:
                if a.ndim == 2 and a.shape[1] == 3 and a.shape[0] >= 8:
                    best_lidar = a
                elif a.ndim == 2 and a.shape[0] == 3 and a.shape[1] >= 8:
                    best_lidar = a.T
                else:
                    # also handle (N,4) (intensity) etc
                    if a.ndim == 2 and a.shape[1] >= 3 and a.shape[0] >= 8:
                        best_lidar = a[:, :3]

    cov_l = 0.0
    cov_d = 0.0
    if best_lidar is not None:
        pts = np.asarray(best_lidar)
        if pts.ndim == 2 and pts.shape[1] >= 3:
            cov_l = _lidar_coverage(pts[:, :3], cfg)
        else:
            best_lidar = None

    if best_depth is not None:
        cam = (sensor_models or {}).get("camera", {})
        hfov = float(cam.get("horizontal_fov_deg", 87.0))
        cov_d = _depth_coverage(np.asarray(best_depth), hfov, cfg)
    else:
        cov_d = 0.0

    if best_lidar is None and best_depth is None:
        return None

    if best_lidar is None and best_depth is not None:
        # depth-only: blind = 1 - depth_coverage
        return float(max(0.0, min(1.0, 1.0 - cov_d)))

    if best_lidar is not None and best_depth is None:
        return float(max(0.0, min(1.0, 1.0 - cov_l)))

    # Fuse modalities with a simple max-coverage: if either perceives a bin, it counts as covered
    # at the scalar level, this is a conservative proxy: coverage_quality ~= max(cov_l, cov_d)
    union_cov = float(max(cov_l, cov_d))
    return float(max(0.0, min(1.0, 1.0 - union_cov)))


def estimate_collision_from_info(info: Any, env_idx: int) -> Optional[float]:
    """
    Return collision_rate for this episode as 0/1 if detectable, else None.
    """
    if info is None:
        return None
    i = _row(info, env_idx)
    if isinstance(i, dict):
        # shallow search for collision-like bool/floats
        for k, v in i.items():
            kl = str(k).lower()
            if "collision" in kl or "contact" in kl:
                if isinstance(v, (bool, np.bool_)):
                    return 1.0 if bool(v) else 0.0
                a = _to_numpy(v)
                if a is not None and np.size(a) == 1:
                    return 1.0 if float(a.reshape(-1)[0]) > 0.5 else 0.0
    return None


def estimate_success_from_info(info: Any, env_idx: int) -> Optional[float]:
    """
    Return success in [0,1] if detectable, else None.
    """
    if info is None:
        return None
    i = _row(info, env_idx)
    if not isinstance(i, dict):
        return None
    for k, v in i.items():
        kl = str(k).lower()
        if any(s in kl for s in ("success", "reached", "goal", "complete")):
            if isinstance(v, (bool, np.bool_)):
                return 1.0 if bool(v) else 0.0
            a = _to_numpy(v)
            if a is not None and np.size(a) == 1:
                return float(max(0.0, min(1.0, float(a.reshape(-1)[0]))))
    return None


@dataclass(frozen=True)
class ForwardRangeConfig:
    """
    Parameters for extracting a *forward-facing* min range (meters) from generic observations.

    Convention: robot forward is +X in the point cloud. For depth images, we use the
    camera center columns as a proxy for "forward" and gate by the camera HFOV.
    """
    # Forward cone in the x-y plane around +X: |atan2(y,x)| <= half_fov_rad
    half_fov_rad: float = math.radians(35.0)  # ~70° total
    max_range: float = 200.0


def min_forward_range_from_obs(
    obs: Any,
    *,
    env_idx: int,
    sensor_models: dict,
    cfg: ForwardRangeConfig | None = None,
) -> Optional[float]:
    """
    Minimum distance to *something* in the forward sector, based on best-effort lidar/points/depth.
    """
    cfg = cfg or ForwardRangeConfig()
    if obs is None:
        return None

    o = _row(obs, env_idx)
    if o is None:
        return None
    o = _unwrap_sensor_obs_dict(o)

    cand: list[float] = []
    hfov_cam = float((sensor_models or {}).get("camera", {}).get("horizontal_fov_deg", 87.0))
    half = float(hfov_cam) * 0.5
    forward_deg = math.degrees(float(cfg.half_fov_rad))

    # 1) Depth: middle rows, only columns that fall inside [−forward_deg, +forward_deg] wrt image center
    dimg: Optional[np.ndarray] = None
    if isinstance(o, dict):
        for k, v in o.items():
            kl = str(k).lower()
            if any(s in kl for s in ("depth", "distance", "z_buf", "zbuf")) and dimg is None:
                a = _to_numpy(v)
                if a is not None and np.asarray(a).size:
                    dimg = np.asarray(a)
    if dimg is not None:
        d = dimg
        while d.ndim > 2 and d.shape[0] == 1:
            d = d[0]
        if d.ndim == 2:
            h, w = d.shape
            h0, h1 = int(0.25 * h), int(0.75 * h)
            band = d[h0:h1, :]
            for col in range(w):
                t = (col + 0.5) / max(w, 1)
                ang_deg = (t - 0.5) * float(hfov_cam)  # [-HFOV/2, +HFOV/2] relative to camera
                if abs(ang_deg) > min(half, forward_deg) + 1e-6:
                    continue
                col_vals = band[:, col]
                col_vals = col_vals[np.isfinite(col_vals)]
                if col_vals.size == 0:
                    continue
                v = float(np.min(col_vals))
                if v > 1e-3 and v < float(cfg.max_range):
                    cand.append(v)

    # 2) 3D points: forward cone in x-y, take planar range
    if isinstance(o, dict):
        for k, v in o.items():
            kl = str(k).lower()
            a = _to_numpy(v)
            if a is None or a.size == 0:
                continue
            a = np.asarray(a, dtype=np.float64)
            if not any(s in kl for s in ("lidar", "point", "pc", "cloud", "range", "returns")):
                continue
            if a.ndim != 2 or a.shape[1] < 3:
                if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] >= 3:
                    a = a.T
            if a.ndim != 2 or a.shape[1] < 3 or a.shape[0] < 1:
                continue
            p = a[:, :3]
            n = np.linalg.norm(p, axis=1)
            p = p[(n > 1e-2) & (n < float(cfg.max_range))]
            if p.size == 0:
                continue
            x = p[:, 0]
            y = p[:, 1]
            azi = np.arctan2(y, x)
            m = (x > 1e-3) & (np.abs(azi) <= float(cfg.half_fov_rad))
            if not bool(np.any(m)):
                continue
            d2 = np.sqrt(np.maximum(x[m] * x[m] + y[m] * y[m], 0.0))
            d2 = d2[np.isfinite(d2)]
            if d2.size:
                cand.append(float(np.min(d2)))

    if not cand:
        return None
    return float(np.min(np.asarray(cand, dtype=np.float64)))


def min_range_any_from_obs(
    obs: Any,
    *,
    env_idx: int,
) -> Optional[float]:
    """
    A looser "minimum range anywhere" from points/lidar/depth, used for clearance checks.
    """
    if obs is None:
        return None
    o = _row(obs, env_idx)
    if o is None:
        return None
    o = _unwrap_sensor_obs_dict(o)

    cand: list[float] = []

    # points
    if isinstance(o, dict):
        for k, v in o.items():
            kl = str(k).lower()
            a = _to_numpy(v)
            if a is None or a.size == 0:
                continue
            a = np.asarray(a, dtype=np.float64)
            if not any(s in kl for s in ("lidar", "point", "pc", "cloud", "range", "returns")):
                continue
            if a.ndim == 2 and a.shape[0] == 3 and a.shape[1] >= 3:
                a = a.T
            if a.ndim != 2 or a.shape[1] < 3:
                continue
            p = a[:, :3]
            n = np.linalg.norm(p, axis=1)
            p = p[(n > 1e-2) & (n < 200.0)]
            if p.size:
                nn = np.linalg.norm(p, axis=1)
                cand.append(float(np.min(nn)))
            break
    if not cand:
        for a in _iter_tensor_leaves(o):
            a = np.asarray(a, dtype=np.float64)
            if a.ndim == 2 and a.shape[1] >= 3 and a.shape[0] >= 3:
                p = a[:, :3]
                n = np.linalg.norm(p, axis=1)
                p = p[n > 1e-2]
                if p.size:
                    cand.append(float(np.min(np.linalg.norm(p.reshape((-1, 3)), axis=1))))
                break

    # depth min
    dimg: Optional[np.ndarray] = None
    if isinstance(o, dict):
        for k, v in o.items():
            kl = str(k).lower()
            if any(s in kl for s in ("depth", "distance", "z_buf", "zbuf")) and dimg is None:
                a = _to_numpy(v)
                if a is not None and np.asarray(a).size:
                    dimg = np.asarray(a)
    if dimg is not None:
        d = dimg
        while d.ndim > 2 and d.shape[0] == 1:
            d = d[0]
        if d.ndim == 2:
            d = d[np.isfinite(d)]
            d = d[(d > 1e-3) & (d < 200.0)]
            if d.size:
                cand.append(float(np.min(d)))

    if not cand:
        return None
    return float(np.min(np.asarray(cand, dtype=np.float64)))


def estimate_contact_int_from_info(info: Any, env_idx: int) -> Optional[int]:
    """
    1/0 contact/collision in this step if the env reports it, else None.
    """
    v = estimate_collision_from_info(info, env_idx)
    if v is None:
        return None
    return 1 if float(v) > 0.5 else 0
