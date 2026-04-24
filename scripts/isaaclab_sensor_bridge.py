#!/usr/bin/env python3
"""
HTTP JSON bridge for connecting this repository's `IsaacSimEvaluator` to an Isaac Lab env.

Implements the env contract used by `sensor_opt.inner_loop.isaac_evaluator.IsaacSimEvaluator`:

  - reconfigure_sensors(env_idx, config, sensor_models)
  - run_rollouts(n_episodes, rng) -> list[EvalMetrics]

This script is meant to be executed with the **Isaac Sim / Isaac Lab python**
(e.g. the Colab setup that installs Isaac Lab under `/usr/local` on common Colab images).

**Google Colab note:** the Jupyter kernel’s `python` may not be the same interpreter as
Isaac/Kit. Launch the bridge with the Isaac install’s Python (often `ISAAC_PYTHON=/usr/local/bin/python`)
in a background `subprocess`, then poll `GET /health` from the notebook. If the HTTP server fails with
`OSError: [Errno 98] Address already in use`, a previous bridge is still bound — stop it (`pkill` / free
the port) before relaunch. Optional env tuning (EULA, Vulkan, Carb) is left to the user’s Colab
notebook. See the repo `README.md` “Google Colab + Isaac Lab” section.

It exposes a tiny HTTP server with POST endpoints:
  - /reconfigure_sensors
  - /run_rollouts

NOTE:
  This repo is sensor-placement focused. The default `--bridge-mode ground` path targets a
  **ground / navigation / legged** Isaac Lab task and computes:
    - `blind_spot_fraction` by fusing *best-effort* 3D point cloud + depth coverage heuristics
    - `collision_rate` and `mean_goal_success` from `info` if present (best-effort)
    with fallbacks to the analytic `fast_baseline_metrics(...)` if observations are not found.

  The `--bridge-mode obstacle` path adds a research-style **static obstacle corridor** *best-effort* workflow:
    - spawns 3-5 `FixedCuboid` obstacles per reset (USD) and re-randomizes positions in a corridor band
    - `t_det_s_p95` (p95 detection latency) from Lidar/Depth/Points forward-sector min range vs `d_warn`
    - `collision_rate` from Isaac Lab `scene.sensors["*contact*"]` when available, else `info` fallbacks
    - `safety_success` / `mean_goal_success` when there is no contact and sensed min range stays > `d_clear`
    (If you need production-grade world coupling, also register obstacles in your task `InteractiveSceneCfg`
    in the Isaac Lab project so resets/domain randomization stay consistent.)

  For real assets, provide USD prim paths for each moved sensor: `sensor_models.<type>.isaac.prim_path`
  (optional `{env_idx}`), or `isaac.mount_prim_paths` keyed by `SingleSensorConfig.slot`, or
  `isaac.prim_paths` (one string per active sensor of that type), or env
  `ISAAC_LIDAR_PRIM` / `ISAAC_CAMERA_PRIM` / `ISAAC_RADAR_PRIM`. Configured poses are re-applied
  after `env.reset()` as well as on `/reconfigure_sensors`, because reset often reloads default transforms.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
import sys
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

import numpy as np


def _add_repo_to_syspath(repo_root: str) -> None:
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def _load_eval_metrics_type(repo_root: str):
    _add_repo_to_syspath(repo_root)
    from sensor_opt.loss.loss import EvalMetrics  # noqa: WPS433 (runtime import on purpose)

    return EvalMetrics


def _euler_rpy_deg_to_quat_wxyz(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """
    RPY in degrees, intrinsic XYZ, quaternion (w, x, y, z) for `omni.isaac.core.prims.XFormPrim`.
    """
    r = math.radians(float(roll_deg))
    p = math.radians(float(pitch_deg))
    y = math.radians(float(yaw_deg))
    cr, sr = math.cos(r * 0.5), math.sin(r * 0.5)
    cp, sp = math.cos(p * 0.5), math.sin(p * 0.5)
    cy, sy = math.cos(y * 0.5), math.sin(y * 0.5)
    w = cr * cp * cy - sr * sp * sy
    x = sr * cp * cy + cr * sp * sy
    y_ = cr * sp * cy - sr * cp * sy
    z = cr * cp * sy + sr * sp * cy
    return (w, x, y_, z)


def _format_prim_path_tmpl(tmpl: str, *, env_idx: int) -> str:
    s = str(tmpl)
    try:
        return s.format(env_idx=env_idx, env=env_idx)
    except (KeyError, ValueError, IndexError):
        return s


def _resolve_isaac_prim_path(
    meta: dict,
    sensor_type: str,
    slot: str,
    type_index: int,
    env_idx: int,
) -> Optional[str]:
    """
    Pick a single USD path for a sensor, in order:
    - meta['mount_prim_paths'][slot] (string or {env_idx} template)
    - meta['prim_paths'][type_index] for multiple prims of the same type
    - meta['prim_path']
    - env ISAAC_<SENSOR_TYPE>_PRIM  (e.g. ISAAC_LIDAR_PRIM), optional {env_idx}
    """
    if not isinstance(meta, dict):
        return None
    mpp = meta.get("mount_prim_paths")
    if isinstance(mpp, dict) and str(slot) in mpp and mpp.get(str(slot)):
        return _format_prim_path_tmpl(str(mpp[str(slot)]), env_idx=env_idx)
    pl = meta.get("prim_paths")
    if isinstance(pl, list) and 0 <= int(type_index) < len(pl) and pl[int(type_index)]:
        return _format_prim_path_tmpl(str(pl[int(type_index)]), env_idx=env_idx)
    pp = meta.get("prim_path")
    if pp:
        return _format_prim_path_tmpl(str(pp), env_idx=env_idx)
    key = f"ISAAC_{str(sensor_type).upper()}_PRIM"
    if os.environ.get(key):
        return _format_prim_path_tmpl(str(os.environ[key]), env_idx=env_idx)
    return None


def _p95(x: list[float]) -> float:
    if not x:
        return 0.0
    a = np.asarray(x, dtype=np.float64)
    return float(np.percentile(a, 95.0))


def _noise_range_m(
    x: Optional[float], rng: np.random.Generator, sigma_m: float
) -> Optional[float]:
    """Additive i.i.d. Gaussian on scalar range-style observations (meters)."""
    if x is None or sigma_m <= 0.0:
        return x
    return float(x) + float(rng.normal(0.0, float(sigma_m)))


def _try_get_physics_dt(env: Any) -> Optional[float]:
    # Isaac Lab / Isaac Sim: best-effort; fall back to CLI sim dt
    for path in (
        "unwrapped.physics_dt",
        "unwrapped.cfg.sim.dt",
        "unwrapped.cfg.sim.physics_dt",
        "unwrapped.sim.get_physics_dt",
    ):
        cur: Any = env
        ok = True
        for part in path.split(".")[1:]:
            if not hasattr(cur, part):
                ok = False
                break
            cur = getattr(cur, part)
        if not ok:
            continue
        try:
            if callable(cur):
                v = float(cur())
            else:
                v = float(cur)
            if v > 0.0 and math.isfinite(v):
                return v
        except Exception:
            continue
    return None


def _set_obstacle_pose(*, path: str, x: float, y: float, z: float) -> None:
    from omni.isaac.core.prims import XFormPrim  # type: ignore

    xf = XFormPrim(str(path))
    xf.set_local_pose(translation=(float(x), float(y), float(z)))


def _ensure_fixed_cuboid(*, path: str, size: float) -> bool:
    """
    Best-effort static collider prim. Returns False if we cannot create it in this sim build.
    """
    try:
        from omni.isaac.core.utils.prims import is_prim_path_valid  # type: ignore
    except Exception:
        is_prim_path_valid = None  # type: ignore[assignment]

    if is_prim_path_valid is not None and bool(is_prim_path_valid(str(path))):
        return True
    try:
        from omni.isaac.core.objects import FixedCuboid  # type: ignore
    except Exception:
        return False
    try:
        # Create at origin; pose is set afterwards by `_set_obstacle_pose`
        _ = FixedCuboid(
            prim_path=str(path),
            name=str(path).split("/")[-1],
            size=float(size),
            position=(0.0, 0.0, 0.0),
        )
        return True
    except Exception:
        return False


def _ensure_obstacles_for_env(
    *,
    env_idx: int,
    n_obst: int,
    size_m: float,
    root: str,
    rng: np.random.Generator,
) -> list[str]:
    """
    Create (once) and place N static cuboids. Shared across all envs if you only have a single
    sim world; for multi-env rendering, you likely need per-env USD roots (set --obstacle-root).
    """
    n = int(max(1, n_obst))
    base = f"{str(root).rstrip('/')}/env_{int(env_idx)}/obstacles"
    prim_paths: list[str] = []
    for i in range(n):
        p = f"{base}/box_{i}"
        prim_paths.append(p)
        # Create once: if it already exists, creation will error; treat as ok.
        ok = _ensure_fixed_cuboid(path=p, size=float(size_m))
        if not ok:
            # try again next time; user may fix Isaac API availability
            continue
    # place
    for i, p in enumerate(prim_paths):
        x = float(rng.uniform(2.0, 8.0))
        y = float(rng.uniform(-2.0, 2.0))
        z = float(size_m) * 0.5
        try:
            _set_obstacle_pose(path=p, x=x, y=y, z=z)
        except Exception:
            pass
    return prim_paths


def _try_contact_flag_from_isaac(*, env: Any, env_idx: int) -> Optional[int]:
    """
    If the env/scene already includes an Isaac Lab contact sensor, read it.
    """
    # Common pattern: env.unwrapped.scene.sensors["contact"] (name varies)
    if not hasattr(env, "unwrapped"):
        return None
    u = env.unwrapped
    scene = getattr(u, "scene", None)
    if scene is None:
        return None
    sensors = getattr(scene, "sensors", None)
    if not isinstance(sensors, dict) or not sensors:
        return None
    for _name, sens in sensors.items():
        n = str(_name).lower()
        if "contact" not in n:
            continue
        data = getattr(sens, "data", None)
        if data is None:
            continue
        for attr in (
            "net_forces_w",
            "force_matrix_w",
            "force_matrix_w_history",
        ):
            if not hasattr(data, attr):
                continue
            a = getattr(data, attr)
            try:
                import torch  # type: ignore

                if torch.is_tensor(a):
                    t = a[env_idx] if a.ndim >= 1 and a.shape[0] > env_idx else a
                    v = float(torch.linalg.norm(t.reshape(-1)[:3]).detach().cpu().item())
                else:
                    ar = np.asarray(a)
                    t2 = ar[env_idx] if ar.ndim >= 1 and ar.shape[0] > env_idx else ar
                    v = float(np.linalg.norm(np.asarray(t2).reshape(-1)[:3]))
            except Exception:
                continue
            if math.isfinite(v) and v > 1e-2:
                return 1
    return None


def _sensor_config_from_dict(payload: dict, SensorConfig, SingleSensorConfig):
    sensors = [SingleSensorConfig(**s) for s in (payload or {}).get("sensors", [])]
    return SensorConfig(sensors=sensors)


def _json_loads_body(req: BaseHTTPRequestHandler) -> dict:
    length = int(req.headers.get("Content-Length", "0") or "0")
    raw = req.rfile.read(length) if length > 0 else b"{}"
    return json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else raw)


def _json_response(req: BaseHTTPRequestHandler, code: int, obj: dict) -> None:
    data = json.dumps(obj).encode("utf-8")
    req.send_response(code)
    req.send_header("Content-Type", "application/json")
    req.send_header("Content-Length", str(len(data)))
    req.end_headers()
    req.wfile.write(data)


class _BridgeState:
    def __init__(
        self,
        num_envs: int,
        SensorConfig,
        SingleSensorConfig,
        EvalMetrics,
        env: Any,
        simulation_app: Any,
        max_steps: int,
        *,
        bridge_mode: str = "generic",
        repo_root: str = "",
        sim_dt: float = 0.02,
        d_warn: float = 3.0,
        d_clear: float = 0.5,
        obstacle_root: str = "/World/bridge_corridor",
        n_obstacles: int = 4,
        obstacle_size_m: float = 0.5,
        sensor_noise_std: float = 0.0,
    ):
        self.num_envs = int(num_envs)
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.SensorConfig = SensorConfig
        self.SingleSensorConfig = SingleSensorConfig
        self.EvalMetrics = EvalMetrics
        self.env = env
        self.simulation_app = simulation_app
        self.max_steps = int(max_steps)
        self.bridge_mode = str(bridge_mode)
        self.repo_root = str(repo_root or "")
        self._warned_no_perception: bool = False
        self._warned_no_usd: bool = False
        self._sim_dt = float(sim_dt) if sim_dt and float(sim_dt) > 0.0 else 0.02
        self._physics_dt: Optional[float] = _try_get_physics_dt(env) or None
        self.d_warn = float(d_warn)
        self.d_clear = float(d_clear)
        self.obstacle_root = str(obstacle_root)
        self.n_obstacles = int(max(3, min(5, int(n_obstacles))))  # clamp 3..5
        self.obstacle_size_m = float(obstacle_size_m)
        # Default i.i.d. noise on range observations (meters). Overridable per /run_rollouts.
        self._default_sensor_noise = float(max(0.0, sensor_noise_std))

        self._configs: list[SensorConfig] = [SensorConfig(sensors=[]) for _ in range(self.num_envs)]
        self._models: list[dict] = [{} for _ in range(self.num_envs)]
        self._warned_vec_reset: bool = False
        self._warned_prim: set[str] = set()

    def reconfigure_sensors(self, env_idx: int, config: dict, sensor_models: dict) -> None:
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(f"env_idx out of range: {env_idx}")
        self._configs[env_idx] = _sensor_config_from_dict(config, self.SensorConfig, self.SingleSensorConfig)
        self._models[env_idx] = dict(sensor_models or {})
        # Apply to USD sensor prims (and again after each `env.reset()` during rollouts).
        self._apply_sensor_config(env_idx=env_idx, config=self._configs[env_idx], sensor_models=self._models[env_idx])

    def _reset_for_row(self, env_idx: int, seed: int) -> Any:
        """
        Vectorized Isaac Lab: try to reset only one env id; if not possible, fall back
        to a global reset (and warn once).
        """
        e = self.env
        u = e.unwrapped if hasattr(e, "unwrapped") else e

        # 1) Single-env is always global reset
        if int(self.num_envs) == 1:
            return e.reset()

        # 2) If API exposes an env_id/env_ids arg, use it
        f = getattr(u, "reset", None)
        if callable(f):
            try:
                sig = inspect.signature(f)
                params = sig.parameters
                kwargs: dict[str, Any] = {}
                if "seed" in params:
                    kwargs["seed"] = int(seed)
                if "env_ids" in params:
                    import torch  # type: ignore

                    kwargs["env_ids"] = torch.tensor([int(env_idx)], device=getattr(u, "device", "cpu"))
                    return f(**kwargs) if kwargs else f()
                if "envs_ids" in params:  # occasional typo/alias in forks
                    import torch  # type: ignore

                    kwargs["envs_ids"] = torch.tensor([int(env_idx)], device=getattr(u, "device", "cpu"))
                    return f(**kwargs) if kwargs else f()
            except TypeError:
                # fall through
                pass
            except Exception:
                # fall through
                pass

        if not self._warned_vec_reset:
            print(
                "[isaaclab_sensor_bridge] num_envs>1 but no per-env reset hook found; "
                "using global `env.reset()` (metrics for different env_idx can be correlated). "
                "For clean isolation, implement/upgrade your task reset to support `env_ids=...`."
            )
            self._warned_vec_reset = True
        return e.reset()

    def run_rollouts(
        self, n_episodes: int, seed: int, sensor_noise_std: Optional[float] = None
    ) -> list[dict[str, float]]:
        noise = self._default_sensor_noise if sensor_noise_std is None else float(sensor_noise_std)
        noise = float(max(0.0, noise))
        out: list[dict[str, float]] = []
        for env_idx in range(self.num_envs):
            m = self._rollout_one_env(
                env_idx=env_idx, n_episodes=int(n_episodes), seed=int(seed), sensor_noise_std=noise
            )
            d = asdict(m)
            # ensure JSON-friendly scalars
            for k, v in list(d.items()):
                if isinstance(v, (bool, np.bool_)):
                    d[k] = bool(v)
                else:
                    d[k] = int(v) if k == "n_episodes" else float(v)
            out.append(d)  # type: ignore[assignment]
        return out

    # --- user-editable region -------------------------------------------------

    def _reapply_sensor_config(self, env_idx: int) -> None:
        """Re-apply the last /reconfigure_sensors config (needed after `env.reset()` reverts prims)."""
        self._apply_sensor_config(
            env_idx=env_idx,
            config=self._configs[env_idx],
            sensor_models=self._models[env_idx],
        )

    def _apply_sensor_config(self, env_idx: int, config, sensor_models: dict) -> None:
        """
        Move sensor `Xform` prims in USD from `SingleSensorConfig` (offsets + pitch/yaw).

        Paths (first match):
          - `sensor_models.<type>.isaac.mount_prim_paths` (per mounting slot)
          - `...isaac.prim_paths` (one entry per *active* sensor of that type, 0-based)
          - `...isaac.prim_path`  (use `{env_idx}` in the string for vectorized envs)
          - env `ISAAC_LIDAR_PRIM` / `ISAAC_CAMERA_PRIM` / `ISAAC_RADAR_PRIM` if YAML is empty
        """
        _add_repo_to_syspath(self.repo_root)

        try:
            from omni.isaac.core.prims import XFormPrim  # type: ignore
        except Exception as e:
            if not self._warned_no_usd:
                print(f"[isaaclab_sensor_bridge] USD API not available; skipping prim moves. ({e})")
                self._warned_no_usd = True
            return

        per_type: dict[str, int] = {}
        for s in config.active_sensors():
            t = s.sensor_type
            idx = int(per_type.get(t, 0))
            per_type[t] = idx + 1

            model = (sensor_models or {}).get(s.sensor_type, {}) or {}
            meta = (model.get("isaac") or {}) if isinstance(model, dict) else {}
            prim_path_f = _resolve_isaac_prim_path(
                meta if isinstance(meta, dict) else {},
                str(s.sensor_type),
                str(s.slot),
                idx,
                int(env_idx),
            )
            if not prim_path_f:
                k = f"missing_path:{s.sensor_type}@{s.slot}#{idx}"
                if k not in self._warned_prim:
                    print(
                        "[isaaclab_sensor_bridge] No prim path for "
                        f"sensor type={s.sensor_type!r} slot={s.slot!r} i={idx} (env {env_idx}). "
                        f"Set `isaac.prim_path` (or `mount_prim_paths` / `prim_paths`, or env ISAAC_"
                        f"{str(s.sensor_type).upper()}_PRIM) so the bridge can move the sensor; "
                        "metrics will not vary between candidates if prims are not moved."
                    )
                    self._warned_prim.add(k)
                continue

            wxyz = _euler_rpy_deg_to_quat_wxyz(0.0, float(s.pitch_deg), float(s.yaw_deg))
            q = np.asarray(wxyz, dtype=np.float64)

            try:
                xf = XFormPrim(str(prim_path_f))
                xf.set_local_pose(
                    translation=(float(s.x_offset), float(s.y_offset), float(s.z_offset)),
                    orientation=q,
                )
            except Exception as e0:
                # Some Isaac versions differ on orientation dtype/shape, or are translation-only.
                try:
                    XFormPrim(str(prim_path_f)).set_local_pose(
                        translation=(float(s.x_offset), float(s.y_offset), float(s.z_offset))
                    )
                except Exception as e2:
                    k = f"err:{prim_path_f}"
                    if k not in self._warned_prim:
                        print(
                            f"[isaaclab_sensor_bridge] set_local_pose failed for {prim_path_f!r} "
                            f"(with orientation: {e0}; translation-only: {e2})"
                        )
                        self._warned_prim.add(k)
        return

    def _rollout_one_env(
        self, env_idx: int, n_episodes: int, seed: int, sensor_noise_std: float = 0.0
    ):
        """
        Run `n_episodes` episodes in the *vectorized* Isaac Lab env, aggregating a scalar
        `EvalMetrics` for the `env_idx` row.

        This is intentionally conservative: it tries `env.step` in the typical Isaac Lab
        way, but the metric mapping is a placeholder.
        """
        if self.env is None:
            raise RuntimeError("Isaac env is not initialized")

        n_eps = max(1, int(n_episodes))
        if str(self.bridge_mode) == "obstacle":
            return self._rollout_one_env_obstacle_risk(
                env_idx=env_idx, n_episodes=n_eps, seed=int(seed), sensor_noise_std=float(sensor_noise_std)
            )
        success_rates: list[float] = []
        coll_rates: list[float] = []
        blind_rates: list[float] = []
        _add_repo_to_syspath(self.repo_root)
        from sensor_opt.inner_loop.baseline_metrics import fast_baseline_metrics  # noqa: WPS433
        from sensor_opt.inner_loop import isaaclab_ground_robot as gr  # noqa: WPS433

        for _ in range(n_eps):
            # Reset: best-effort *per env row* if the Isaac env supports it; otherwise global reset.
            # Keep torch RNG loosely tied to the request seed (best-effort determinism).
            import torch  # type: ignore

            torch.manual_seed(int(seed) & 0xFFFF_FFFF)
            rng = np.random.default_rng(int(seed) & 0xFFFF_FFFF)

            reset_out = self._reset_for_row(env_idx, int(seed))
            if isinstance(reset_out, tuple) and len(reset_out) >= 1:
                obs = reset_out[0]
            else:
                obs = reset_out
            # `reset()` can restore default USD sensor poses; re-apply the candidate config.
            self._reapply_sensor_config(env_idx)
            t = 0
            ep_reward = 0.0
            ep_blinds: list[float] = []
            had_collision: Optional[float] = None
            had_success: Optional[float] = None
            # Roll until done or cap steps. If env doesn't expose a simple done signal,
            # we just run for max_steps.
            while t < self.max_steps and bool(self.simulation_app.is_running()):
                action = self._sample_action()
                step_out = self.env.step(action)

                # Common patterns:
                #   obs, reward, terminated, truncated, info = env.step
                #   obs, reward, dones, info = env.step
                if isinstance(step_out, tuple) and len(step_out) >= 3:
                    reward = float(self._get_row(step_out[1], env_idx))
                    done = self._is_done_from_step_out(step_out, env_idx=env_idx)
                else:
                    reward = 0.0
                    done = False
                info = step_out[4] if isinstance(step_out, tuple) and len(step_out) >= 5 else None
                if obs is None and isinstance(step_out, tuple) and len(step_out) >= 1:
                    obs = step_out[0]

                ep_reward += reward

                if self.bridge_mode == "ground":
                    b = gr.estimate_blind_spot_fraction_from_obs(
                        obs, env_idx=env_idx, sensor_models=self._models[env_idx]
                    )
                    if b is not None:
                        ep_blinds.append(float(b))
                    c = gr.estimate_collision_from_info(info, env_idx=env_idx)
                    if c is not None and had_collision is None:
                        had_collision = float(c)
                    s = gr.estimate_success_from_info(info, env_idx=env_idx)
                    if s is not None and had_success is None:
                        had_success = float(s)

                t += 1
                obs = step_out[0] if isinstance(step_out, tuple) and len(step_out) >= 1 else None
                if done:
                    break

            # finalize episode metrics
            cfg0 = self._configs[env_idx]
            sm0 = self._models[env_idx]

            if self.bridge_mode == "ground":
                if ep_blinds:
                    v = float(np.mean(ep_blinds))
                    # Same noise scale as t_det / ranges: map meters -> [0,1] jitter (documented in YAML)
                    if float(sensor_noise_std) > 0.0:
                        v = float(
                            np.clip(
                                v + float(rng.normal(0.0, float(sensor_noise_std) * 0.1)),
                                0.0,
                                1.0,
                            )
                        )
                    blind_rates.append(v)
                else:
                    base = fast_baseline_metrics(
                        cfg0,
                        sm0,
                        n_episodes=1,
                        rng=rng,
                        noise_std=0.0,
                    )
                    blind_rates.append(float(base.blind_spot_fraction))
                    if not self._warned_no_perception:
                        print(
                            "[isaaclab_sensor_bridge][ground] Could not find lidar/depth-like observations; "
                            "using analytic baseline for blind spot this episode."
                        )
                        self._warned_no_perception = True

                if had_collision is not None:
                    coll_rates.append(float(had_collision))
                else:
                    base = fast_baseline_metrics(
                        cfg0,
                        sm0,
                        n_episodes=1,
                        rng=rng,
                        noise_std=0.0,
                    )
                    coll_rates.append(float(base.collision_rate))

                if had_success is not None:
                    success_rates.append(float(had_success))
                else:
                    # best-effort reward-based proxy for navigation/velocity tasks
                    success_rates.append(float(np.tanh(max(0.0, ep_reward) / 5.0)))
            else:
                # generic: keep the older lightweight proxy
                success_proxy = float(np.tanh(max(0.0, ep_reward) / 50.0))
                success_rates.append(float(np.clip(success_proxy, 0.0, 1.0)))
                coll_rates.append(0.0)
                blind_rates.append(0.0)

        return self.EvalMetrics(
            collision_rate=float(np.mean(coll_rates)) if coll_rates else 0.0,
            blind_spot_fraction=float(np.mean(blind_rates)) if blind_rates else 0.0,
            mean_goal_success=float(np.mean(success_rates)) if success_rates else 0.0,
            n_episodes=n_eps,
        )

    def _rollout_one_env_obstacle_risk(
        self, env_idx: int, n_episodes: int, seed: int, sensor_noise_std: float = 0.0
    ):
        """
        Obstacle-corridor / safety metrics:
        - Spawns 3-5 static cuboids per reset (see `_ensure_obstacles_for_env`)
        - t_det: first time min forward range < d_warn
        - Contact/collision: ContactSensor in scene (best-effort) + `info` fallbacks
        - Success: (no contact) AND (min proximity > d_clear) over episode
        """
        _add_repo_to_syspath(self.repo_root)
        from sensor_opt.inner_loop import isaaclab_ground_robot as gr  # noqa: WPS433

        dt = float(self._physics_dt) if self._physics_dt is not None else float(self._sim_dt)
        t_max = float(self.max_steps) * float(dt)
        d_warn = float(self.d_warn)
        d_clear = float(self.d_clear)
        sigma_m = float(max(0.0, sensor_noise_std))

        t_dets: list[float] = []
        coll: list[float] = []
        safety: list[float] = []
        blind_proxy: list[float] = []

        for ep in range(int(n_episodes)):
            import torch  # type: ignore

            torch.manual_seed((int(seed) + int(env_idx) * 100_003 + int(ep) * 1_000_003) & 0xFFFF_FFFF)
            erng = np.random.default_rng(
                (int(seed) + int(env_idx) * 100_003 + int(ep) * 1_000_003) & 0xFFFF_FFFF
            )
            ep_seed = (int(seed) + int(env_idx) * 100_003 + int(ep) * 1_000_003) & 0xFFFF_FFFF

            reset_out = self._reset_for_row(env_idx, int(ep_seed))
            if isinstance(reset_out, tuple) and len(reset_out) >= 1:
                obs = reset_out[0]
            else:
                obs = reset_out
            self._reapply_sensor_config(env_idx)
            # Randomize 3-5 obstacles *after* reset (scene stable), per project spec
            n_obst = int(erng.integers(3, 6))
            _ = _ensure_obstacles_for_env(
                env_idx=env_idx,
                n_obst=n_obst,
                size_m=float(self.obstacle_size_m),
                root=str(self.obstacle_root),
                rng=erng,
            )
            t = 0
            t_det: Optional[float] = None
            had_contact = 0
            min_glob: Optional[float] = None
            # blind proxy: reuse legacy blind-spot heuristic to keep a comparable perception term
            ep_blind: list[float] = []

            g0 = gr.min_range_any_from_obs(obs, env_idx=env_idx)
            if g0 is not None:
                g0n = _noise_range_m(float(g0), erng, sigma_m)
                min_glob = g0n
            b0 = gr.estimate_blind_spot_fraction_from_obs(
                obs, env_idx=env_idx, sensor_models=self._models[env_idx]
            )
            if b0 is not None:
                bj = float(b0)
                if sigma_m > 0.0:
                    bj = float(np.clip(bj + float(erng.normal(0.0, sigma_m * 0.1)), 0.0, 1.0))
                ep_blind.append(bj)
            fr0 = gr.min_forward_range_from_obs(
                obs, env_idx=env_idx, sensor_models=self._models[env_idx]
            )
            if fr0 is not None:
                fr0 = _noise_range_m(float(fr0), erng, sigma_m)
            if t_det is None and fr0 is not None and float(fr0) < d_warn - 1e-6:
                t_det = 0.0

            while t < self.max_steps and bool(self.simulation_app.is_running()):
                action = self._sample_action()
                step_out = self.env.step(action)
                if isinstance(step_out, tuple) and len(step_out) >= 1:
                    obs = step_out[0]
                info = step_out[4] if isinstance(step_out, tuple) and len(step_out) >= 5 else None

                fr = gr.min_forward_range_from_obs(
                    obs, env_idx=env_idx, sensor_models=self._models[env_idx]
                )
                if fr is not None:
                    fr = _noise_range_m(float(fr), erng, sigma_m)
                g = gr.min_range_any_from_obs(obs, env_idx=env_idx)
                if g is not None:
                    gn = _noise_range_m(float(g), erng, sigma_m)
                    if gn is not None:
                        min_glob = gn if min_glob is None else float(min(float(min_glob), float(gn)))

                b = gr.estimate_blind_spot_fraction_from_obs(
                    obs, env_idx=env_idx, sensor_models=self._models[env_idx]
                )
                if b is not None:
                    bj = float(b)
                    if sigma_m > 0.0:
                        bj = float(np.clip(bj + float(erng.normal(0.0, sigma_m * 0.1)), 0.0, 1.0))
                    ep_blind.append(bj)

                # t_det: first time forward range is below d_warn
                if t_det is None and fr is not None and float(fr) < d_warn - 1e-6:
                    t_det = float(t) * float(dt)

                c_isaac = _try_contact_flag_from_isaac(env=self.env, env_idx=env_idx)
                c_info = gr.estimate_contact_int_from_info(info, env_idx)
                c = c_isaac if c_isaac is not None else c_info
                if c == 1:
                    had_contact = 1

                done = self._is_done_from_step_out(step_out, env_idx=env_idx) if isinstance(step_out, tuple) else False
                t += 1
                if bool(done) or had_contact == 1:
                    break

            if t_det is None:
                # penalty for no detection: treat as "late" detection at episode horizon
                t_det = float(t_max) if t_max > 0.0 else float(self.max_steps) * float(dt)
            t_dets.append(float(t_det))

            coll.append(float(had_contact))
            ok = (had_contact == 0) and (min_glob is not None) and (float(min_glob) > d_clear)
            safety.append(1.0 if ok else 0.0)
            blind_proxy.append(float(np.mean(ep_blind)) if ep_blind else 0.0)

        t_det_p95 = _p95(t_dets)
        t_det_mean = float(np.mean(np.asarray(t_dets, dtype=np.float64))) if t_dets else 0.0
        return self.EvalMetrics(
            collision_rate=float(np.mean(coll)) if coll else 0.0,
            # keep blind_spot slot populated with a perception proxy for analysis / compatibility
            blind_spot_fraction=float(np.mean(blind_proxy)) if blind_proxy else 0.0,
            # surface safety success in mean_goal_success for easy downstream use
            mean_goal_success=float(np.mean(safety)) if safety else 0.0,
            n_episodes=int(n_episodes),
            t_det_s=float(t_det_mean),
            t_det_s_p95=float(t_det_p95),
            episode_time_s=float(t_max),
            safety_success=float(np.mean(safety)) if safety else 0.0,
        )

    def _sample_action(self):
        # Follow Isaac Lab's `scripts/environments/random_agent.py` for tensor actions.
        # NOTE: This samples the *entire* vectorized action tensor (all envs). If you need
        # per-row control, set actions[env_idx, ...] to your command while keeping a valid
        # shape for the env.
        import torch  # type: ignore

        dev = self.env.unwrapped.device  # Isaac Lab env exposes a torch device
        with torch.inference_mode():
            shape = self.env.action_space.shape
            return 2 * torch.rand(shape, device=dev) - 1

    def _get_row(self, reward_obj: Any, env_idx: int) -> float:
        # try torch / numpy / list scalar extraction
        try:
            import torch  # type: ignore

            if torch.is_tensor(reward_obj):
                r = reward_obj[env_idx]
                return float(r.detach().cpu().reshape(-1)[0].item())
        except Exception:
            pass

        try:
            if hasattr(reward_obj, "cpu"):
                r = reward_obj[env_idx]
                return float(np.asarray(r).reshape(-1)[0])
        except Exception:
            pass

        try:
            arr = np.asarray(reward_obj)
            if arr.size > env_idx:
                return float(arr.reshape(-1)[env_idx])
        except Exception:
            pass

        return float(reward_obj)

    def _is_done_from_step_out(self, step_out: tuple, env_idx: int) -> bool:
        if len(step_out) >= 5:
            term = step_out[2]
            trunc = step_out[3]
            return bool(self._get_row_or_bool(term, env_idx) or self._get_row_or_bool(trunc, env_idx))
        if len(step_out) >= 4:
            dones = step_out[3]
            return self._get_row_or_bool(dones, env_idx)
        return False

    def _get_row_or_bool(self, obj: Any, env_idx: int) -> bool:
        try:
            import torch  # type: ignore

            if torch.is_tensor(obj):
                v = obj[env_idx]
                return bool(v.detach().cpu().reshape(-1)[0].item() > 0.5)
        except Exception:
            pass
        try:
            arr = np.asarray(obj)
            if arr.size > env_idx:
                return bool(arr.reshape(-1)[env_idx])
        except Exception:
            pass
        return bool(obj)


class _HandlerFactory:
    def __init__(self, state: _BridgeState):
        self.state = state

    def __call__(self, *args, **kwargs):
        return _BridgeRequestHandler(self.state, *args, **kwargs)


class _BridgeRequestHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, state: _BridgeState, *args, **kwargs):
        self._state = state
        super().__init__(*args, **kwargs)

    def do_GET(self) -> None:  # noqa: N802
        # Minimal readiness endpoint for Colab/CI polling.
        if self.path in ("/health", "/health/"):
            return _json_response(self, 200, {"ok": True, "service": "isaaclab_sensor_bridge"})
        return _json_response(self, 404, {"error": f"unknown path: {self.path}"})

    def do_POST(self) -> None:  # noqa: N802
        try:
            body = _json_loads_body(self)
            if self.path == "/reconfigure_sensors":
                self._state.reconfigure_sensors(
                    env_idx=int(body["env_idx"]),
                    config=dict(body.get("config", {})),
                    sensor_models=dict(body.get("sensor_models", {})),
                )
                return _json_response(self, 200, {"ok": True})
            if self.path == "/run_rollouts":
                n_episodes = int(body.get("n_episodes", 1))
                seed = int(body.get("seed", 0))
                if "sensor_noise_std" in body:
                    sn = float(body["sensor_noise_std"])
                    metrics = self._state.run_rollouts(
                        n_episodes=n_episodes, seed=seed, sensor_noise_std=sn
                    )
                else:
                    metrics = self._state.run_rollouts(n_episodes=n_episodes, seed=seed)
                return _json_response(self, 200, {"metrics": metrics})
            return _json_response(self, 404, {"error": f"unknown path: {self.path}"})
        except Exception as e:
            return _json_response(self, 500, {"error": str(e), "type": e.__class__.__name__})

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # keep stdout quieter in Colab
        return


def main() -> int:
    p = argparse.ArgumentParser(description="HTTP JSON bridge: Isaac Lab <-> sensor_placement_opt IsaacSimEvaluator")
    p.add_argument("--host", type=str, default="127.0.0.1", help="HTTP bind address (use 127.0.0.1 for Colab local bridge).")
    p.add_argument("--port", type=int, default=8010, help="HTTP port for /reconfigure_sensors and /run_rollouts")
    p.add_argument("--max-steps", type=int, default=500, help="Per-episode cap (prevents runaway loops).")
    p.add_argument("--repo-root", type=str, default="", help="Path to the `sensor_placement_opt` git checkout (enables `import sensor_opt`).")
    p.add_argument("--video", action="store_true", help="Record rollout videos via gymnasium RecordVideo wrapper.")
    p.add_argument("--video_length", type=int, default=200, help="Number of frames per recorded video clip.")
    p.add_argument("--video_interval", type=int, default=1, help="Record every N episodes (1 = every episode).")
    p.add_argument(
        "--bridge-mode",
        type=str,
        default="ground",
        choices=["ground", "generic", "obstacle"],
        help=(
            "ground: mobile/navigation-style metric fusion; "
            "obstacle: static-obstacle corridor + t_det / contact / safety success; "
            "generic: old lightweight proxy (CartPole-like)."
        ),
    )
    p.add_argument(
        "--sim-dt",
        type=float,
        default=0.02,
        help="Fallback sim dt (sec/step) if we cannot read dt from the Isaac env (used for t_det).",
    )
    p.add_argument("--d-warn", type=float, default=3.0, help="Warning distance d_warn (meters) for t_det.")
    p.add_argument(
        "--d-clear",
        type=float,
        default=0.5,
        help="Safety clearance: success requires min sensed range > d_clear (meters) and no contact.",
    )
    p.add_argument(
        "--obstacle-root",
        type=str,
        default="/World/bridge_corridor",
        help="USD root path for bridge-spawned obstacles (per-env: .../env_{i}/obstacles/box_k).",
    )
    p.add_argument(
        "--n-obstacles",
        type=int,
        default=4,
        help="Number of static obstacles to spawn (clamped to 3..5; each episode re-randomizes count in that range).",
    )
    p.add_argument(
        "--obstacle-size",
        type=float,
        default=0.5,
        help="Approximate cuboid size in meters (passed to omni.isaac.core.objects.FixedCuboid).",
    )
    p.add_argument(
        "--sensor-noise-std",
        dest="sensor_noise_std",
        type=float,
        default=0.0,
        help=(
            "Std dev of i.i.d. Gaussian noise on range-like observations (meters). "
            "The optimizer can override per request via JSON sensor_noise_std on /run_rollouts. "
            "Environment variable SENSOR_NOISE_STD overrides this default when set."
        ),
    )

    # Import after argparse so the script is importable in non-Isaac dev machines.
    try:
        from isaaclab.app import AppLauncher  # type: ignore
    except Exception as e:
        print(
            "ERROR: `isaaclab` is not importable. Run this script with the Isaac Sim / Isaac Lab python.\n"
            f"Import error: {e}",
            file=sys.stderr,
        )
        return 1

    # Isaac Lab standard flags (headless, device, task, num_envs, etc.)
    p.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    p.add_argument("--task", type=str, default=None, help="Name of the Isaac Lab task.")
    p.add_argument("--disable_fabric", action="store_true", default=False, help="Disable Fabric (Isaac Lab standard flag).")
    AppLauncher.add_app_launcher_args(p)
    args, unknown = p.parse_known_args()

    if unknown:
        print(f"[isaaclab_sensor_bridge] Ignoring unknown args: {unknown}", file=sys.stderr)

    sensor_noise_bridge = float(getattr(args, "sensor_noise_std", 0.0) or 0.0)
    if os.environ.get("SENSOR_NOISE_STD"):
        try:
            sensor_noise_bridge = float(os.environ["SENSOR_NOISE_STD"])
        except ValueError:
            pass

    if not args.task:
        # Reasonable defaults for Colab + Isaac Lab 2.1 common demos:
        # - ANYmal-C navigation (ground + nav objective)
        # - CartPole (classic smoke test; used by many tutorials)
        if args.bridge_mode in ("ground", "obstacle"):
            args.task = "Isaac-Navigation-Flat-Anymal-C-v0"
        else:
            args.task = "Isaac-Cartpole-v0"
    if args.num_envs is None:
        args.num_envs = 1

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Heavy imports (must be after app launch, per Isaac Lab guidance)
    import gymnasium as gym  # noqa: WPS433
    import isaaclab_tasks  # noqa: F401, WPS433 (register envs)
    from isaaclab_tasks.utils import parse_env_cfg  # type: ignore

    _add_repo_to_syspath(args.repo_root)
    from sensor_opt.encoding.config import SensorConfig, SingleSensorConfig  # noqa: WPS433

    EvalMetrics = _load_eval_metrics_type(args.repo_root)

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=int(args.num_envs),
        use_fabric=not bool(args.disable_fabric),
    )
    if args.enable_cameras:
        # Common Isaac Lab pattern for camera-based tasks
        try:
            env_cfg.sim.render.enable_camera_views = True
        except Exception:
            pass

    env = gym.make(args.task, cfg=env_cfg)
    if bool(args.video):
        from gymnasium.wrappers.record_video import RecordVideo  # noqa: WPS433 (runtime import in Isaac runtime)

        video_dir = os.environ.get("ISAAC_VIDEO_DIR", "/tmp/isaaclab_video")
        os.makedirs(video_dir, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            episode_trigger=lambda ep_id: int(ep_id) % max(1, int(args.video_interval)) == 0,
            name_prefix=f"bridge_{args.bridge_mode}",
            video_length=int(args.video_length),
        )
        print(f"[isaaclab_sensor_bridge] Video recording ON → {video_dir}", flush=True)

    state = _BridgeState(
        num_envs=int(args.num_envs),
        SensorConfig=SensorConfig,
        SingleSensorConfig=SingleSensorConfig,
        EvalMetrics=EvalMetrics,
        env=env,
        simulation_app=simulation_app,
        max_steps=int(args.max_steps),
        bridge_mode=str(args.bridge_mode),
        repo_root=str(args.repo_root),
        sim_dt=float(args.sim_dt),
        d_warn=float(args.d_warn),
        d_clear=float(args.d_clear),
        obstacle_root=str(args.obstacle_root),
        n_obstacles=int(args.n_obstacles),
        obstacle_size_m=float(args.obstacle_size),
        sensor_noise_std=sensor_noise_bridge,
    )

    httpd = ThreadingHTTPServer((args.host, int(args.port)), _HandlerFactory(state))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"[isaaclab_sensor_bridge] listening on http://{args.host}:{args.port}")
    print(
        f"[isaaclab_sensor_bridge] task={args.task!r} num_envs={int(args.num_envs)} "
        f"bridge_mode={args.bridge_mode!r} default_sensor_noise_std_m={sensor_noise_bridge!r}"
    )

    # Block forever in main thread; user stops with interrupt (Colab) or process kill
    try:
        while simulation_app.is_running():
            # tiny sleep; avoid busy loop
            import time

            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            env.close()
        except Exception:
            pass
        try:
            simulation_app.close()
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
