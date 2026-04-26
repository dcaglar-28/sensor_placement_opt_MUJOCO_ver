"""
MuJoCo: kinematic vehicle on +X, five fixed sites, mocap obstacle pool.
Contacts vehicle–obstacles are excluded in MJCF. Geometry via mj_kinematics (no mj_step in episode).
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.loss.loss import EvalMetrics
from sensor_opt.simulation.mjcf import SLOT_NAMES, build_vehicle_mjcf
from sensor_opt.simulation.mujoco_runner import run_episode
from sensor_opt.simulation.obstacles import generate_obstacles, get_generation_seed
from sensor_opt.simulation.sensor_specs import get_sensor_specs

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None  # type: ignore[assignment]


def _slot_to_config(
    config: SensorConfig,
) -> Dict[str, str]:
    out: Dict[str, str] = {s: "disabled" for s in SLOT_NAMES}
    for s in config.sensors:
        if s.slot in out:
            out[str(s.slot)] = str(s.sensor_type) if s.is_active() else "disabled"
    return out


class MujocoEnvManager:
    def __init__(
        self,
        *,
        num_envs: int = 1,
        n_obstacles: int = 10,
        path_length_m: float = 20.0,
        vehicle_speed_mps: float = 2.0,
        timestep_s: float = 0.02,
        base_random_seed: int = 42,
        max_steps_per_episode: int = 2000,
        _sensor_noise_std: float = 0.0,
        **_: Any,
    ) -> None:
        if mujoco is None:  # pragma: no cover
            raise ImportError("MuJoCo is not installed. Install with: pip install mujoco>=3.1")
        self.num_envs = int(max(1, num_envs))
        self.n_obstacles = int(max(1, n_obstacles))
        self.path_length_m = float(path_length_m)
        self.vehicle_speed_mps = float(vehicle_speed_mps)
        self.timestep_s = float(timestep_s)
        self.base_random_seed = int(base_random_seed)
        self.max_steps_per_episode = int(max_steps_per_episode)
        self._sn = float(_sensor_noise_std or 0.0)
        self._slots: list[tuple[SensorConfig | None, dict | None]] = [
            (None, None) for _ in range(self.num_envs)
        ]
        self._gen: int = 0
        xml = build_vehicle_mjcf(self.n_obstacles)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self._exp_cfg: dict = {}

    def set_experiment_config(self, cfg: dict) -> None:
        self._exp_cfg = dict(cfg or {})
        il = self._exp_cfg.get("inner_loop", {})
        mj: Dict[str, Any] = (il or {}).get("mujoco", {}) or {} if isinstance(il, dict) else {}
        if isinstance(mj, dict) and mj:
            self.n_obstacles = int(mj.get("n_obstacles", self.n_obstacles))
            self.path_length_m = float(mj.get("path_length_m", self.path_length_m))
            self.vehicle_speed_mps = float(mj.get("vehicle_speed_mps", self.vehicle_speed_mps))
            self.timestep_s = float(mj.get("timestep_s", self.timestep_s))
            if mj.get("base_random_seed") is not None:
                self.base_random_seed = int(mj.get("base_random_seed", 42))
        rt = self._exp_cfg.get("runtime", {})
        if not isinstance(rt, dict):
            rt = {}
        if rt.get("base_random_seed") is not None:
            self.base_random_seed = int(rt["base_random_seed"])
        if rt.get("n_obstacles") is not None:
            self.n_obstacles = int(rt["n_obstacles"])
        if rt.get("path_length_m") is not None:
            self.path_length_m = float(rt["path_length_m"])
        if rt.get("vehicle_speed_mps") is not None:
            self.vehicle_speed_mps = float(rt["vehicle_speed_mps"])
        if rt.get("timestep_s") is not None:
            self.timestep_s = float(rt["timestep_s"])

    def reconfigure_sensors(self, env_idx: int, config: SensorConfig, sensor_models: dict) -> None:
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError("env_idx out of range")
        self._slots[env_idx] = (config, dict(sensor_models))

    def run_rollouts(
        self,
        n_episodes: int,
        rng: np.random.Generator,
        sensor_noise_std: float = 0.0,
        generation: int = 0,
    ) -> list[EvalMetrics]:
        _ = (rng, sensor_noise_std)  # episode seeds derived from gen + base
        self._gen = int(generation)
        out: list[EvalMetrics] = []
        for i in range(self.num_envs):
            cfg, sm = self._slots[i]
            if cfg is None or sm is None:
                out.append(
                    _zero_metrics(n_episodes, self.n_obstacles)
                )
                continue
            if not cfg.active_sensors():
                out.append(
                    _zero_metrics(n_episodes, self.n_obstacles)
                )
                continue
            out.append(self._rollout_one(cfg, sm, n_episodes))
        return out

    def _rollout_one(self, config: SensorConfig, sensor_models: dict, n_episodes: int) -> EvalMetrics:
        scfg = _slot_to_config(config)
        sspec = get_sensor_specs(self._exp_cfg)
        n_ep = max(1, n_episodes)
        gen_seed = get_generation_seed(self._gen, self.base_random_seed)
        sim_c = {
            "path_length_m": self.path_length_m,
            "vehicle_speed_mps": self.vehicle_speed_mps,
            "timestep_s": self.timestep_s,
            "n_obstacles": self.n_obstacles,
        }
        covs: list[float] = []
        ndets: list[float] = []
        mdd: list[float] = []
        ftm: list[float] = []
        pslots: list[dict] = []
        n_obs_l: list[int] = []
        t_ep: list[float] = []
        for ep in range(n_ep):
            pos = generate_obstacles(
                self.n_obstacles,
                self.path_length_m,
                gen_seed + ep,
            )
            m = run_episode(
                self.model,
                self.data,
                scfg,
                pos,
                sim_c,
                sspec,
                np.random.default_rng(int(gen_seed) + 17 * ep),
            )
            covs.append(float(m["coverage_fraction"]))
            ndets.append(float(m["n_detected"]))
            mdd.append(float(m["mean_detection_distance"]))
            ft = m["first_detection_times"]
            T = float(m["episode_duration_s"])
            t_ep.append(T)
            vals: list[float] = []
            for f in ft:
                if f is None:
                    vals.append(T)
                else:
                    vals.append(float(f))
            ftm.append(float(np.mean(vals)) if vals else T)
            pslots.append(dict(m.get("per_slot_first_hits", {})))
            n_obs_l.append(int(m.get("n_obstacles", self.n_obstacles)))
        pmean = {s: 0.0 for s in SLOT_NAMES}
        for p in pslots:
            for k, v in p.items():
                if k in pmean:
                    pmean[k] += float(v) / n_ep
        return EvalMetrics(
            collision_rate=0.0,
            blind_spot_fraction=0.0,
            mean_goal_success=float(np.mean(ndets) / max(float(self.n_obstacles), 1.0)),
            n_episodes=n_ep,
            t_det_s=float(np.mean(ftm)),
            t_det_s_p95=float(np.percentile(np.asarray(ftm, dtype=np.float64), 95.0)),
            episode_time_s=float(np.mean(t_ep)) if t_ep else 0.0,
            detection_miss_rate=1.0 - float(np.mean(ndets) / max(float(self.n_obstacles), 1.0)),
            coverage_fraction=float(np.mean(covs)),
            n_detected=float(np.mean(ndets)),
            n_obstacles=float(np.mean(n_obs_l)),
            mean_detection_distance_m=float(np.mean(mdd)) if mdd else 0.0,
            first_detection_time_mean=float(np.mean(ftm)),
            per_slot_first_hits={k: float(v) for k, v in pmean.items()},
        )


def _zero_metrics(
    n_episodes: int, n_obstacles: int
) -> EvalMetrics:
    T = 1.0
    return EvalMetrics(
        collision_rate=0.0,
        blind_spot_fraction=1.0,
        mean_goal_success=0.0,
        n_episodes=max(0, n_episodes),
        episode_time_s=T,
        coverage_fraction=0.0,
        n_detected=0.0,
        n_obstacles=float(n_obstacles),
        mean_detection_distance_m=0.0,
        first_detection_time_mean=T,
        per_slot_first_hits={s: 0.0 for s in SLOT_NAMES},
    )
