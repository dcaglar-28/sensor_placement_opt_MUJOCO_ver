# Sensor Placement Optimization
**Bi-Level CMA-ES Search for Autonomous Robot Sensor Configurations**

## Architecture

```
Outer Loop (CMA-ES)          Inner Loop (Isaac Sim / Mock Isaac)
─────────────────────        ──────────────────────────────
encode config → σ-vector  →  load env + attach sensors
mutate population         ←  run K=15 episodes
evaluate L = α·col +         return {collision_rate,
         β·blind +                    blind_spot_fraction,
         γ·cost                       sensor_cost}
select + update CMA
log to CSV / MLflow
```

## Quick Start (CPU-only, no Isaac Sim needed)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m sensor_opt.run_experiment --config configs/default.yaml --dummy
```

Notes:
- `--dummy` is an **alias** for the built-in `mock_isaac` evaluator (no Isaac Sim required).
- To run explicitly via config, set `inner_loop.mode: mock_isaac`.

## Google Colab + Isaac Lab (optional)

The entry notebook is `notebooks/sensor_opt_isaaclab_colab.ipynb` (unofficial [Isaac Sim / Isaac Lab on Colab](https://github.com/j3soon/isaac-sim-colab) install scripts). A lighter HTTP-only walkthrough (mock Isaac) is in `notebooks/sensor_placement_opt_colab.ipynb`.

**Typical flow (as documented in `sensor_opt_isaaclab_colab.ipynb` — the notebook is the source of truth; edit `REPO_URL` there to match your fork):**

1. **Install** Isaac Sim + Isaac Lab using the `wget` + `bash install-*.sh` cells (long-running, GPU runtime recommended).
2. **Clone** this repo to `/content/sensor_placement_opt` and `pip install -r requirements.txt` with **`sys.executable -m pip`** so packages land in the Colab kernel.
3. **NumPy** — After heavy installs, some Colab stacks pull **NumPy 2.x**; this project expects **NumPy 1.26.x**. The notebook uses **`sys.executable -m pip`** to reinstall **`numpy<2.0.0`**, then you **restart the runtime** and re-run the clone cell so imports pick up 1.26.x.
4. **Bridge process** — Run `scripts/isaaclab_sensor_bridge.py` with **Isaac’s Python** (commonly `/usr/local/bin/python` on these Colab images), *not* the Jupyter kernel. The bridge cell passes **`--enable-cameras`** so tasks can expose `depth` / point-like data for **ground**-mode heuristics. It polls **`http://127.0.0.1:<port>/health`** (the notebook’s default **port** is **8010**; keep it the same in the bridge and optimizer cells).
5. **Environment variables (Colab / bridge)** — Set as needed: **`ISAAC_TASK`**, **`BRIDGE_MODE`** (`ground` or `obstacle`), **`MAX_STEPS`**. In **obstacle** mode the bridge can also use **`D_WARN`**, **`D_CLEAR`**, **`SIM_DT`**. The optimizer cell picks **`CONFIG_PATH`** from **`CONFIG_PATH`**, or defaults to `configs/default.yaml` vs `configs/obstacle_isaaclab.yaml` from **`BRIDGE_MODE`**.
6. **Modes** — **`ground`**: `blind_spot_fraction` heuristics. **`obstacle`**: corridor metrics (`t_det_s_p95`, contact collision rate, safety fields) — pair with `configs/obstacle_isaaclab.yaml` and `loss.mode: obstacle_latency`.
7. **JSON client** — Map the full `/run_rollouts` row with `eval_metrics_from_bridge_row()` so optional keys (`t_det_s_p95`, `safety_success`, …) are not dropped.
8. **Port / process debugging** — If you see **`OSError: [Errno 98] Address already in use`**, stop old bridge processes (`pkill` / free the port). The notebook also includes optional helper cells (port check, **`JAX_PLATFORMS=cpu`**, `health` probe, `ps`, `nvidia-smi`, `tail` of `/tmp/isaaclab_sensor_bridge.log`) for the same default port.
9. **USD sensor prims** — The bridge applies each candidate to **`sensor_models.<type>.isaac.prim_path`** (or **`mount_prim_paths` / `prim_paths`**, or env **`ISAAC_LIDAR_PRIM`**, **`ISAAC_CAMERA_PRIM`**, **`ISAAC_RADAR_PRIM`**) and **re-applies after every `env.reset()`**, because resets often restore default transforms. If no path is set, all candidates can look identical (e.g. `blind_spot_fraction` stuck at 1.0). The notebook’s interactive cell sets env vars the bridge subprocess inherits when you launch with `subprocess` without replacing `os.environ`.

## Project Structure

```
sensor_placement_opt/
├── notebooks/          # Colab walkthroughs (Isaac Lab + mock HTTP)
├── sensor_opt/
│   ├── encoding/       # Encode/decode sensor configs ↔ float vectors
│   ├── loss/           # L = α·collision + β·blind_spot + γ·cost
│   ├── cma/            # CMA-ES outer loop wrapper
│   ├── inner_loop/     # Isaac Sim evaluator + mock (CPU) evaluator
│   └── logging/        # CSV + MLflow experiment tracking
├── configs/            # YAML experiment configs
├── tests/              # Unit tests (pytest)
└── results/            # Auto-created run outputs
```

## Phases

| Phase | Status | Requires |
|-------|--------|----------|
| 0 — Scaffold + mock eval | CPU-only | Python 3.10+, cma |
| 1 — Isaac Sim integration | Isaac Sim 4.x, GPU |
| 2 — Full RL inner loop | Phase 1 + PPO agent |

**Loss modes (Isaac / research):** `configs/default.yaml` uses `loss.mode: default` (collision + blind spot + cost). For obstacle-corridor evaluation + \( \alpha\cdot p95(t_{det}) + \beta\cdot\text{collision} \), use `configs/obstacle_isaaclab.yaml` (`loss.mode: obstacle_latency`).