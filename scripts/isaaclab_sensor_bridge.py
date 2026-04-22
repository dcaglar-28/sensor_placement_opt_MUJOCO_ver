#!/usr/bin/env python3
"""
HTTP JSON bridge for connecting this repository's `IsaacSimEvaluator` to an Isaac Lab env.

Implements the env contract used by `sensor_opt.inner_loop.isaac_evaluator.IsaacSimEvaluator`:

  - reconfigure_sensors(env_idx, config, sensor_models)
  - run_rollouts(n_episodes, rng) -> list[EvalMetrics]

This script is meant to be executed with the **Isaac Sim / Isaac Lab python**
(e.g. the Colab setup that installs Isaac Lab under `/content/IsaacLab`).

It exposes a tiny HTTP server with POST endpoints:
  - /reconfigure_sensors
  - /run_rollouts

NOTE:
  This repo is sensor-placement focused. The default `--bridge-mode ground` path targets a
  **ground / navigation / legged** Isaac Lab task and computes:
    - `blind_spot_fraction` by fusing *best-effort* 3D point cloud + depth coverage heuristics
    - `collision_rate` and `mean_goal_success` from `info` if present (best-effort)
    with fallbacks to the analytic `fast_baseline_metrics(...)` if observations are not found.

  For real assets, you should still provide USD prim paths in `sensor_models.<type>.isaac.prim_path`
  (and optional `{env_idx}` formatting) so `_apply_sensor_config` can move sensors.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
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

        self._configs: list[SensorConfig] = [SensorConfig(sensors=[]) for _ in range(self.num_envs)]
        self._models: list[dict] = [{} for _ in range(self.num_envs)]

    def reconfigure_sensors(self, env_idx: int, config: dict, sensor_models: dict) -> None:
        if env_idx < 0 or env_idx >= self.num_envs:
            raise IndexError(f"env_idx out of range: {env_idx}")
        self._configs[env_idx] = _sensor_config_from_dict(config, self.SensorConfig, self.SingleSensorConfig)
        self._models[env_idx] = dict(sensor_models or {})
        # Hook point for the user: apply to Isaac scene / sensor prims.
        self._apply_sensor_config(env_idx=env_idx, config=self._configs[env_idx], sensor_models=self._models[env_idx])

    def run_rollouts(self, n_episodes: int, seed: int) -> list[dict[str, float]]:
        out: list[dict[str, float]] = []
        for env_idx in range(self.num_envs):
            m = self._rollout_one_env(env_idx=env_idx, n_episodes=int(n_episodes), seed=int(seed))
            out.append(
                {
                    "collision_rate": float(m.collision_rate),
                    "blind_spot_fraction": float(m.blind_spot_fraction),
                    "mean_goal_success": float(m.mean_goal_success),
                    "n_episodes": int(m.n_episodes),
                }
            )
        return out

    # --- user-editable region -------------------------------------------------

    def _apply_sensor_config(self, env_idx: int, config, sensor_models: dict) -> None:
        """
        Best-effort USD `Xform` updates if prim paths are provided in YAML:
          sensor_models:
            lidar:
              isaac: { prim_path: "/World/envs/env_{env_idx}/.../Lidar" }
        """
        _add_repo_to_syspath(self.repo_root)

        for s in config.active_sensors():
            model = (sensor_models or {}).get(s.sensor_type, {}) or {}
            meta = (model.get("isaac") or {}) if isinstance(model, dict) else {}
            if not isinstance(meta, dict):
                continue
            prim_path = meta.get("prim_path", None)
            if not prim_path:
                continue
            try:
                prim_path_f = str(prim_path).format(env_idx=env_idx)
            except Exception:
                prim_path_f = str(prim_path)

            try:
                from omni.isaac.core.prims import XFormPrim  # type: ignore
            except Exception as e:
                if not self._warned_no_usd:
                    print(f"[isaaclab_sensor_bridge] USD API not available; skipping prim moves. ({e})")
                    self._warned_no_usd = True
                return

            try:
                xf = XFormPrim(prim_path_f)
                # Translation offsets in meters; rotation hooks can be added if you expose euler in USD
                xf.set_local_pose(translation=(float(s.x_offset), float(s.y_offset), float(s.z_offset)))
            except Exception as e:
                if not self._warned_no_usd:
                    print(f"[isaaclab_sensor_bridge] Failed to set pose for {prim_path_f!r}: {e}")
                    self._warned_no_usd = True
        return

    def _rollout_one_env(self, env_idx: int, n_episodes: int, seed: int):
        """
        Run `n_episodes` episodes in the *vectorized* Isaac Lab env, aggregating a scalar
        `EvalMetrics` for the `env_idx` row.

        This is intentionally conservative: it tries `env.step` in the typical Isaac Lab
        way, but the metric mapping is a placeholder.
        """
        if self.env is None:
            raise RuntimeError("Isaac env is not initialized")

        n_eps = max(1, int(n_episodes))
        success_rates: list[float] = []
        coll_rates: list[float] = []
        blind_rates: list[float] = []
        _add_repo_to_syspath(self.repo_root)
        from sensor_opt.inner_loop.baseline_metrics import fast_baseline_metrics  # noqa: WPS433
        from sensor_opt.inner_loop import isaaclab_ground_robot as gr  # noqa: WPS433

        for _ in range(n_eps):
            # Reset the full vectorized env (Isaac Lab envs are commonly vec).
            # Keep torch RNG loosely tied to the request seed (best-effort determinism).
            import torch  # type: ignore

            torch.manual_seed(int(seed) & 0xFFFF_FFFF)
            rng = np.random.default_rng(int(seed) & 0xFFFF_FFFF)

            reset_out = self.env.reset()
            if isinstance(reset_out, tuple) and len(reset_out) >= 1:
                obs = reset_out[0]
            else:
                obs = reset_out
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
                    blind_rates.append(float(np.mean(ep_blinds)))
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
    p.add_argument(
        "--bridge-mode",
        type=str,
        default="ground",
        choices=["ground", "generic"],
        help="ground: mobile/navigation-style metric fusion; generic: old lightweight proxy (CartPole-like).",
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

    if not args.task:
        # Reasonable defaults for Colab + Isaac Lab 2.1 common demos:
        # - ANYmal-C navigation (ground + nav objective)
        # - CartPole (classic smoke test; used by many tutorials)
        if args.bridge_mode == "ground":
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
    )

    httpd = ThreadingHTTPServer((args.host, int(args.port)), _HandlerFactory(state))
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    print(f"[isaaclab_sensor_bridge] listening on http://{args.host}:{args.port}")
    print(
        f"[isaaclab_sensor_bridge] task={args.task!r} num_envs={int(args.num_envs)} "
        f"bridge_mode={args.bridge_mode!r}"
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
