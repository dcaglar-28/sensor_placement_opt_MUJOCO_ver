"""
HTTP JSON client for `scripts/isaaclab_sensor_bridge.py` (Colab / local).

Implements the env contract used by `IsaacSimEvaluator`: `reconfigure_sensors` and
`run_rollouts`. Passes `sensor_noise_std` in the /run_rollouts body when the
bridge supports it (see `inner_loop/isaac_sim.sensor_noise_std` in YAML).
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict
from typing import List

import numpy as np

from sensor_opt.encoding.config import SensorConfig
from sensor_opt.loss.loss import EvalMetrics


def eval_metrics_from_bridge_row(m: dict) -> EvalMetrics:
    return EvalMetrics(
        collision_rate=float(m["collision_rate"]),
        blind_spot_fraction=float(m.get("blind_spot_fraction", 0.0)),
        mean_goal_success=float(m.get("mean_goal_success", 0.0)),
        n_episodes=int(m["n_episodes"]),
        t_det_s=float(m.get("t_det_s", 0.0)),
        t_det_s_p95=float(m.get("t_det_s_p95", 0.0)),
        episode_time_s=float(m.get("episode_time_s", 0.0)),
        safety_success=float(m.get("safety_success", 0.0)),
    )


class BridgeJsonClient:
    def __init__(self, base_url: str, timeout_sec: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = float(timeout_sec)

    def reconfigure_sensors(self, env_idx: int, config: SensorConfig, sensor_models: dict) -> None:
        payload = {"env_idx": int(env_idx), "config": asdict(config), "sensor_models": sensor_models}
        self._post_json("/reconfigure_sensors", payload)

    def run_rollouts(
        self,
        n_episodes: int,
        rng: np.random.Generator,
        sensor_noise_std: float = 0.0,
    ) -> List[EvalMetrics]:
        seed = int(rng.integers(0, 2**31 - 1))
        payload = {
            "n_episodes": int(n_episodes),
            "seed": int(seed),
            "sensor_noise_std": float(sensor_noise_std),
        }
        data = self._post_json("/run_rollouts", payload)
        return [eval_metrics_from_bridge_row(m) for m in data["metrics"]]

    def _post_json(self, path: str, payload: dict) -> dict:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_sec) as r:
            return json.loads(r.read().decode("utf-8"))
