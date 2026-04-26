"""
MuJoCo inner loop: `reconfigure_sensors` + `run_rollouts` -> `list[EvalMetrics]`.

* **Prism** box vehicle (planar: slide X, Y + yaw), body frame +X forward, +Y left, +Z up.
* **Multiple mocap cylinders** spawn at random xy each episode; contacts vs prism.
* **Perception (geometric)**: range + FOV cone from each active mount. Used for
  (optional) goal coverage and for **hazard detection time**: time until each
  obstacle was first seen, then t_det = max over obstacles (last hazard to come
  into FOV+range). **detection_miss_rate** = fraction of episodes where
  collision/timeout occurred before all obstacles were first seen.

See `loss.mode: mujoco_tri` for **speed** (p95 t_det) · **accuracy** (miss rate) ·
**cost** (sensor budget in loss).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from sensor_opt.encoding.config import SensorConfig, SingleSensorConfig
from sensor_opt.inner_loop.prism_path_scene import prism_sensor_local_translations_m
from sensor_opt.loss.loss import EvalMetrics

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]

# Non-prism “slot” names: boresight angle in the body horizontal plane (rad), +X = 0.
SLOT_BORESIGHT_RAD: Dict[str, float] = {
    "front": 0.0,
    "rear": math.pi,
    "left": 0.5 * math.pi,
    "right": -0.5 * math.pi,
    "top": 0.0,
    "front-left": 0.25 * math.pi,
    "front-right": -0.25 * math.pi,
    "rear-left": 0.75 * math.pi,
    "rear-right": -0.75 * math.pi,
}


def _boresight_local_prism(mount: str) -> np.ndarray:
    m = str(mount)
    if m in ("prism_front_face_l", "prism_front_face_r"):
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if m in ("prism_top_edge_l", "prism_top_edge_r"):
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if m == "prism_left_edge":
        return np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if m == "prism_right_edge":
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    return np.array([1.0, 0.0, 0.0], dtype=np.float64)


def _boresight_local_generic(slot: str) -> np.ndarray:
    ang = float(SLOT_BORESIGHT_RAD.get(str(slot), 0.0))
    c, s = math.cos(ang), math.sin(ang)
    return np.array([c, s, 0.0], dtype=np.float64)


def _rotz_mat(rad: float) -> np.ndarray:
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def build_prism_mjcf_n(sx: float, sy: float, sz: float, n_obstacles: int) -> str:
    """Prism + N mocap cylinder obstacles (randomized positions in rollout)."""
    sx, sy, sz = float(sx), float(sy), float(sz)
    n_obstacles = int(max(1, n_obstacles))
    z_body = sz + 0.04
    parts: List[str] = [
        f"""
<mujoco model="prism_sensor_opt">
  <option timestep="0.02" gravity="0 0 -9.81" integrator="RK4" cone="elliptic"/>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" pos="0 0 0" size="10 10 0.01" rgba="0.72 0.72 0.72 1"/>
    <body name="prism" pos="0 0 {z_body}">
      <joint name="prism_jx" type="slide" axis="1 0 0" range="-7 7" damping="0.4"/>
      <joint name="prism_jy" type="slide" axis="0 1 0" range="-7 7" damping="0.4"/>
      <joint name="prism_jrz" type="hinge" axis="0 0 1" range="-3.14159 3.14159" damping="0.2"/>
      <geom name="prism_geom" type="box" size="{sx} {sy} {sz}" mass="4" rgba="0.2 0.45 0.85 1"/>
    </body>
"""
    ]
    for i in range(n_obstacles):
        parts.append(
            f"""    <body name="obst_{i}" mocap="true" pos="0 0 {z_body}">
      <geom name="obst_geom_{i}" type="cylinder" size="0.2 0.25" rgba="0.85 0.2 0.2 1"/>
    </body>
"""
        )
    parts.append("  </worldbody>\n</mujoco>\n")
    return "".join(parts)


def _rotmat_body(data, body_id: int) -> np.ndarray:
    xm = np.asarray(data.xmat, dtype=np.float64).ravel()
    n = 9 * int(body_id)
    if n + 9 <= xm.size:
        return xm[n : n + 9].reshape(3, 3)
    M = np.asarray(data.xmat[body_id], dtype=np.float64)
    if M.shape == (3, 3):
        return M
    if M.size == 9:
        return M.reshape(3, 3)
    return np.eye(3, dtype=np.float64)


def _xpos3(data, body_id: int) -> np.ndarray:
    xp = np.asarray(data.xpos, dtype=np.float64).ravel()
    n = 3 * int(body_id)
    if n + 3 <= xp.size:
        return np.array(xp[n : n + 3], dtype=np.float64)
    v = np.asarray(data.xpos[body_id], dtype=np.float64)
    return v.ravel()[:3].copy()


def _sensor_sees_point_3d(
    p_mount: np.ndarray,
    d_boresight: np.ndarray,
    target: np.ndarray,
    *,
    meta: dict,
    s: SingleSensorConfig,
    rng: np.random.Generator,
    noise_std: float,
) -> bool:
    r_base = float(meta.get("range_m", 5.0)) * float(s.range_fraction)
    r_max = r_base + float(rng.normal(0.0, noise_std))
    v = target - p_mount
    dist = float(np.linalg.norm(v))
    if r_max <= 0.0 or dist > r_max + 1e-7:
        return False
    if dist < 1e-9:
        return True
    u = v / dist
    d = d_boresight / (np.linalg.norm(d_boresight) + 1e-9)
    hfov = float(meta.get("horizontal_fov_deg", 90.0)) * float(s.hfov_fraction)
    vfov = float(meta.get("vertical_fov_deg", hfov)) * float(s.hfov_fraction)
    if hfov >= 359.0:
        return True
    cos_ang = float(np.clip(np.dot(d, u), -1.0, 1.0))
    ang = float(math.acos(cos_ang))
    h_lim = 0.5 * math.radians(max(1.0, hfov))
    v_lim = 0.5 * math.radians(max(1.0, vfov))
    lim = float(math.sqrt(0.5 * (h_lim**2 + v_lim**2)))
    return ang <= lim + 1e-6


def _mount_world_frames(
    R: np.ndarray,
    p_com: np.ndarray,
    sensors: List[SingleSensorConfig],
    sensor_models: dict,
    sx: float,
    sy: float,
    sz: float,
) -> List[Tuple[SingleSensorConfig, np.ndarray, np.ndarray, dict]]:
    out: list[tuple[SingleSensorConfig, np.ndarray, np.ndarray, dict]] = []
    mtab = prism_sensor_local_translations_m(sx, sy, sz)
    for s in sensors:
        meta = sensor_models.get(s.sensor_type, {})
        slot = str(s.slot)
        if slot in mtab:
            p_l = np.array(mtab[slot], dtype=np.float64)
            d_l = _boresight_local_prism(slot)
        else:
            p_l = np.zeros(3, dtype=np.float64)
            d_l = _boresight_local_generic(slot)
        d_l = _rotz_mat(math.radians(float(s.yaw_deg))) @ d_l
        d_l = d_l / (np.linalg.norm(d_l) + 1e-9)
        p_m = p_com + R @ p_l
        d_w = R @ d_l
        d_w = d_w / (np.linalg.norm(d_w) + 1e-9)
        out.append((s, p_m, d_w, meta))
    return out


def _geom_coverage_goal(
    R: np.ndarray,
    p_com: np.ndarray,
    goal: np.ndarray,
    sensors: List[SingleSensorConfig],
    sensor_models: dict,
    sx: float,
    sy: float,
    sz: float,
    rng: np.random.Generator,
    noise_std: float,
) -> float:
    if not sensors:
        return 0.0
    for s, p_m, d_w, meta in _mount_world_frames(R, p_com, sensors, sensor_models, sx, sy, sz):
        if _sensor_sees_point_3d(
            p_m, d_w, goal, meta=meta, s=s, rng=rng, noise_std=noise_std
        ):
            return 1.0
    return 0.0


def _update_obstacle_detections(
    data,
    *,
    prism_bid: int,
    mocap_ids: list[int],
    first_seen: list[float | None],
    t_now: float,
    act: list[SingleSensorConfig],
    sensor_models: dict,
    sx: float,
    sy: float,
    sz: float,
    rng: np.random.Generator,
    noise_std: float,
) -> None:
    R = _rotmat_body(data, prism_bid)
    p_com = _xpos3(data, prism_bid)
    frames = _mount_world_frames(R, p_com, act, sensor_models, sx, sy, sz)
    nobs = len(mocap_ids)
    for i in range(nobs):
        if first_seen[i] is not None:
            continue
        z = float(data.mocap_pos[mocap_ids[i], 2])
        ox = float(data.mocap_pos[mocap_ids[i], 0])
        oy = float(data.mocap_pos[mocap_ids[i], 1])
        pt = np.array([ox, oy, z], dtype=np.float64)
        for s, p_m, d_w, meta in frames:
            if _sensor_sees_point_3d(
                p_m, d_w, pt, meta=meta, s=s, rng=rng, noise_std=noise_std
            ):
                first_seen[i] = t_now
                break


def _check_prism_hits_obstacles(
    model, data, robot_gid: int, obs_gids: list[int]
) -> bool:
    s_obs = set(obs_gids)
    for k in range(data.ncon):
        c = data.contact[k]
        g1, g2 = c.geom1, c.geom2
        if g1 == robot_gid and g2 in s_obs:
            return True
        if g2 == robot_gid and g1 in s_obs:
            return True
    return False


@dataclass
class _SlotState:
    config: SensorConfig | None = None
    sensor_models: dict | None = None


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _p95(x: list[float]) -> float:
    if not x:
        return 0.0
    a = np.asarray(x, dtype=np.float64)
    return float(np.percentile(a, 95.0))


class MujocoEnvManager:
    def __init__(
        self,
        *,
        num_envs: int = 1,
        max_steps_per_episode: int = 500,
        goal_reach_m: float = 0.28,
        max_speed: float = 1.2,
        kp: float = 1.1,
        world_limit: float = 2.2,
        prism_half_extents: Tuple[float, float, float] | list[float] = (0.4, 0.2, 0.12),
        goal_z: float = 0.2,
        n_random_obstacles: int = 4,
        obstacle_xy_limit: float = 2.0,
        _sensor_noise_std: float = 0.0,
    ):
        if mujoco is None:  # pragma: no cover
            raise ImportError("MuJoCo is not installed. Install with: pip install mujoco>=3.1")
        if num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        self.num_envs = int(num_envs)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self.goal_reach_m = float(goal_reach_m)
        self.max_speed = float(max_speed)
        self.kp = float(kp)
        self.world_limit = float(world_limit)
        self.goal_z = float(goal_z)
        h = [float(x) for x in prism_half_extents]
        if len(h) != 3:
            raise ValueError("prism_half_extents must be length-3 (sx, sy, sz) half-sizes in meters")
        self.sx, self.sy, self.sz = h[0], h[1], h[2]
        if min(self.sx, self.sy, self.sz) <= 0.0:
            raise ValueError("prism half-extents must be positive")
        self.n_random_obstacles = int(max(1, n_random_obstacles))
        self.obstacle_xy_limit = float(obstacle_xy_limit)
        self._default_sensor_noise = float(_sensor_noise_std)
        self._slots = [_SlotState() for _ in range(self.num_envs)]

        xml = build_prism_mjcf_n(self.sx, self.sy, self.sz, self.n_random_obstacles)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._sim_dt = float(self.model.opt.timestep)
        self._prism_bid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "prism"))
        jx = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "prism_jx"))
        jy = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "prism_jy"))
        jr = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "prism_jrz"))
        self._qadr = (int(self.model.jnt_qposadr[jx]), int(self.model.jnt_qposadr[jy]), int(self.model.jnt_qposadr[jr]))
        self._dadr = (int(self.model.jnt_dofadr[jx]), int(self.model.jnt_dofadr[jy]), int(self.model.jnt_dofadr[jr]))
        self._mocap_ids: list[int] = []
        self._obs_gids: list[int] = []
        for i in range(self.n_random_obstacles):
            bid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obst_{i}"))
            mid = int(self.model.body_mocapid[bid])
            if mid < 0:
                raise RuntimeError(f"obst_{i} must be mocap in MJCF")
            self._mocap_ids.append(mid)
            gid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obst_geom_{i}"))
            self._obs_gids.append(gid)
        self._prism_gid = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "prism_geom"))

    def reconfigure_sensors(self, env_idx: int, config: SensorConfig, sensor_models: dict) -> None:
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(f"env_idx out of range: {env_idx}")
        self._slots[env_idx].config = config
        self._slots[env_idx].sensor_models = dict(sensor_models)

    def run_rollouts(
        self,
        n_episodes: int,
        rng: np.random.Generator,
        sensor_noise_std: float = 0.0,
    ) -> list[EvalMetrics]:
        noise = float(sensor_noise_std) if sensor_noise_std is not None else 0.0
        if n_episodes <= 0:
            z = EvalMetrics(
                collision_rate=0.0,
                blind_spot_fraction=1.0,
                mean_goal_success=0.0,
                n_episodes=max(0, n_episodes),
                t_det_s=0.0,
                t_det_s_p95=0.0,
                detection_miss_rate=0.0,
            )
            return [z] * self.num_envs
        return [
            self._rollout_one_slot(
                config=self._slots[i].config or SensorConfig(sensors=[]),
                sensor_models=self._slots[i].sensor_models or {},
                n_episodes=n_episodes,
                rng=rng,
                sensor_noise_std=noise,
            )
            for i in range(self.num_envs)
        ]

    def _rollout_one_slot(
        self,
        *,
        config: SensorConfig,
        sensor_models: dict,
        n_episodes: int,
        rng: np.random.Generator,
        sensor_noise_std: float,
    ) -> EvalMetrics:
        act = config.active_sensors()
        n_col = 0
        n_ok = 0
        blinds: list[float] = []
        t_dets: list[float] = []
        misses: list[float] = []
        q0, q1, q2 = self._qadr
        d0, d1, d2 = self._dadr
        bid = self._prism_bid
        n_obs = self.n_random_obstacles
        z_body = self.sz + 0.04
        t_hor = self.max_steps_per_episode * self._sim_dt

        for _ in range(n_episodes):
            mujoco.mj_resetData(self.model, self.data)
            gx = float(rng.uniform(-self.world_limit, self.world_limit))
            gy = float(rng.uniform(-self.world_limit, self.world_limit))
            gz = self.goal_z
            goal = np.array([gx, gy, gz], dtype=np.float64)
            y0 = float(rng.uniform(-math.pi, math.pi))
            self.data.qpos[q0] = 0.0
            self.data.qpos[q1] = 0.0
            self.data.qpos[q2] = y0
            for d in (d0, d1, d2):
                self.data.qvel[d] = 0.0
            for i in range(n_obs):
                ox = float(rng.uniform(-self.obstacle_xy_limit, self.obstacle_xy_limit))
                oy = float(rng.uniform(-self.obstacle_xy_limit, self.obstacle_xy_limit))
                if abs(ox) < 0.2 and abs(oy) < 0.2:
                    ox += 0.5
                self.data.mocap_pos[self._mocap_ids[i], 0] = ox
                self.data.mocap_pos[self._mocap_ids[i], 1] = oy
                self.data.mocap_pos[self._mocap_ids[i], 2] = z_body
            mujoco.mj_forward(self.model, self.data)

            R0 = _rotmat_body(self.data, bid)
            p0 = _xpos3(self.data, bid)
            cover = _geom_coverage_goal(
                R0, p0, goal, act, sensor_models, self.sx, self.sy, self.sz, rng, sensor_noise_std
            )
            blinds.append(clamp01(1.0 - cover))

            first_seen: List[float | None] = [None] * n_obs
            _update_obstacle_detections(
                self.data,
                prism_bid=bid,
                mocap_ids=self._mocap_ids,
                first_seen=first_seen,
                t_now=0.0,
                act=act,
                sensor_models=sensor_models,
                sx=self.sx,
                sy=self.sy,
                sz=self.sz,
                rng=rng,
                noise_std=sensor_noise_std,
            )

            had_contact = False
            reached = False
            for sidx in range(self.max_steps_per_episode):
                com = _xpos3(self.data, bid)
                dxy = np.array([gx - com[0], gy - com[1]], dtype=np.float64)
                dist = float(np.linalg.norm(dxy))
                if _check_prism_hits_obstacles(
                    self.model, self.data, self._prism_gid, self._obs_gids
                ):
                    had_contact = True
                    break
                if dist < self.goal_reach_m and abs(com[2] - gz) < 0.6:
                    reached = True
                    break
                t_now = float(sidx + 1) * self._sim_dt
                _update_obstacle_detections(
                    self.data,
                    prism_bid=bid,
                    mocap_ids=self._mocap_ids,
                    first_seen=first_seen,
                    t_now=t_now,
                    act=act,
                    sensor_models=sensor_models,
                    sx=self.sx,
                    sy=self.sy,
                    sz=self.sz,
                    rng=rng,
                    noise_std=sensor_noise_std,
                )
                if dist < 1e-6:
                    break
                if dist > 0:
                    ux, uy = dxy[0] / dist, dxy[1] / dist
                    sp = min(self.kp * dist, self.max_speed)
                    self.data.qvel[d0] = sp * ux
                    self.data.qvel[d1] = sp * uy
                    self.data.qvel[d2] = 0.0
                mujoco.mj_step(self.model, self.data)

            if all(f is not None for f in first_seen):
                t_ep = max(float(f) for f in first_seen if f is not None)  # type: ignore[misc, arg-type]
            else:
                t_ep = t_hor
            t_dets.append(t_ep)

            if not act:
                miss = 1.0
            else:
                all_seen = all(f is not None for f in first_seen)
                miss = 1.0 if ((had_contact and not all_seen) or (not all_seen and not reached)) else 0.0
            misses.append(miss)

            if had_contact:
                n_col += 1
            if reached and not had_contact:
                n_ok += 1

        mean_t = float(np.mean(t_dets)) if t_dets else 0.0
        p95t = _p95(t_dets) if t_dets else 0.0
        mean_bl = float(np.mean(blinds)) if blinds else 1.0
        det_miss = float(np.mean(misses)) if misses else 1.0
        return EvalMetrics(
            collision_rate=n_col / float(n_episodes),
            blind_spot_fraction=clamp01(mean_bl),
            mean_goal_success=n_ok / float(n_episodes),
            n_episodes=n_episodes,
            t_det_s=mean_t,
            t_det_s_p95=p95t,
            episode_time_s=t_hor,
            detection_miss_rate=clamp01(det_miss),
        )
