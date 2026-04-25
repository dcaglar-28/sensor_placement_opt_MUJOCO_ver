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

Tests (after `pip install -r requirements.txt`):

```bash
python -m pytest tests/ -q
```

## Google Colab + Isaac Lab (optional)

The entry notebook is `notebooks/sensor_opt_isaaclab_colab.ipynb` (unofficial [Isaac Sim / Isaac Lab on Colab](https://github.com/j3soon/isaac-sim-colab) install scripts). A lighter HTTP-only walkthrough (mock Isaac) is in `notebooks/sensor_placement_opt_colab.ipynb`.

**Typical flow (as documented in `sensor_opt_isaaclab_colab.ipynb` — the notebook is the source of truth; edit `REPO_URL` there to match your fork):**

1. **Install** Isaac Sim + Isaac Lab using the `wget` + `bash install-*.sh` cells (long-running, GPU runtime recommended).
2. **Clone** this repo to `/content/sensor_placement_opt` and `pip install -r requirements.txt` with **`sys.executable -m pip`** so packages land in the Colab kernel.
3. **NumPy** — After heavy installs, some Colab stacks pull **NumPy 2.x**; this project expects **NumPy 1.26.x**. The notebook uses **`sys.executable -m pip`** to reinstall **`numpy<2.0.0`**, then you **restart the runtime** and re-run the clone cell so imports pick up 1.26.x.
4. **Bridge process** — Run `scripts/isaaclab_sensor_bridge.py` with **Isaac’s Python** (commonly `/usr/local/bin/python` on these Colab images), *not* the Jupyter kernel. The bridge cell passes **`--enable-cameras`** so tasks can expose `depth` / point-like data for **ground**-mode heuristics. It polls **`http://127.0.0.1:<port>/health`** (the notebook’s default **port** is **8010**; keep it the same in the bridge and optimizer cells).
5. **Environment variables (Colab / bridge)** — Set as needed: **`ISAAC_TASK`**, **`BRIDGE_MODE`** (`ground` or `obstacle`), **`MAX_STEPS`**. In **obstacle** mode the bridge can also use **`D_WARN`**, **`D_CLEAR`**, **`SIM_DT`**, **`OBSTACLE_LAYOUT`** (`corridor` or `prism_path`), **`PRISM_PROTO`** (set to `0` to skip), **`PRISM_PATH_ROOT`**. The optimizer cell picks **`CONFIG_PATH`** from **`CONFIG_PATH`**, or defaults to `configs/default.yaml` vs `configs/obstacle_isaaclab.yaml` from **`BRIDGE_MODE`**.
6. **Modes** — **`ground`**: `blind_spot_fraction` heuristics. **`obstacle`**: static cuboids (3–5 per reset) plus the same **recognition / safety** metrics (`t_det_s` / `t_det_s_p95`, contact / collision, clearance-style success). Use **`loss.mode: obstacle_latency`** in YAML. The default `configs/obstacle_isaaclab.yaml` targets a **rectangular prism** with **six fixed mount points** and **`fixed_mount_order: true`** (each CMA-ES block maps to one physical site—see `sensor_opt/inner_loop/prism_path_scene.py`). Optional **`--enable-prism-prototype`** on the bridge spawns a prototype prism + mount `Xform`s and slides the body along +X; use **`--obstacle-layout prism_path`** for a wider obstacle sampling band. The older many-slot “corridor” config lives in **`configs/obstacle_isaaclab_corridor.yaml`**. The **HTTP bridge** remains the normal split: Jupyter / optimizer process ↔ `isaaclab_sensor_bridge.py` (Isaac’s Python); installing Isaac Lab in Colab does not remove that boundary unless you merge evaluator and sim in one process.
7. **JSON client** — Map the full `/run_rollouts` row with `eval_metrics_from_bridge_row()` so optional keys (`t_det_s_p95`, `safety_success`, …) are not dropped.
8. **Bridge options** — **`--video`** / **`ISAAC_VIDEO_DIR`**: record episodes when supported. **`--sensor-noise-std`** / **`SENSOR_NOISE_STD`**: optional range / heuristic noise (must match what you set in config as `inner_loop.isaac_sim.sensor_noise_std` for fair runs). Each HTTP request can also override noise per rollout.
9. **Port / process debugging** — If you see **`OSError: [Errno 98] Address already in use`**, stop old bridge processes (`pkill` / free the port). The notebook also includes optional helper cells (port check, **`JAX_PLATFORMS=cpu`**, `health` probe, `ps`, `nvidia-smi`, `tail` of `/tmp/isaaclab_sensor_bridge.log`) for the same default port.
10. **USD sensor prims** — The bridge applies each candidate to **`sensor_models.<type>.isaac.prim_path`** (or **`mount_prim_paths`**, keyed by **`SingleSensorConfig.slot`**, or **`prim_paths`**, or env **`ISAAC_LIDAR_PRIM`**, **`ISAAC_CAMERA_PRIM`**, **`ISAAC_RADAR_PRIM`**) and **re-applies after every `env.reset()`**, because resets often restore default transforms. If no path is set, all candidates can look identical (e.g. `blind_spot_fraction` stuck at 1.0). The notebook’s interactive cell sets env vars the bridge subprocess inherits when you launch with `subprocess` without replacing `os.environ`.
11. **Paper figures (notebook)** — After an optimization run, a dedicated cell can render **SVG** plots via `IPython.display.SVG` (no matplotlib in `requirements.txt`): convergence (multi-run overlay), Pareto scatter, CMA-ES **σ** vs generation, correlation heatmap, top-down layout schematic, sample-efficiency curve, and commented stubs for baselines, CDFs, and hypervolume ablations. Point paths at the run directory (parent of `generations.csv`).

## Sensor catalog (YAML)

Configs can list a **`sensor_catalog`** (per-type USD prim paths, FOV, cost, and optional Isaac metadata) and **`sensor_choices` / slot** definitions. At load time, `apply_sensor_catalog()` in `sensor_opt/config/catalog.py` materializes the merged **`sensor_models`** dict used by the encoder and evaluators. This keeps one catalog per hardware family while reusing the same optimization pipeline.

Per-type **inventory** goes in **`sensor_budget`**: `usermax` is how many units of that sensor you *can* install (copied to `max_count` for the encoder if you omit `max_count`). The search can use **any** count from 0 up to that cap; unused capacity stays `disabled` in the encoded layout, so you do not have to use every unit. Optional `min_count` is a floor (e.g. at least one camera). You may set `usermax` and `max_count` to the same integer, or only one of them. Machine hardware specs such as `gpu_cores`, `unified_memory_gb`, and `memory_bandwidth_gbps` are validated only for `inner_loop.mode: isaac_sim`; mock runs do not need them.

## Run outputs and paper-style artifacts

Each CMA-ES run writes under the run directory (next to `generations.csv`):

| File | Purpose |
|------|---------|
| `generations.csv` | Per-generation `best_loss`, `mean_loss`, `std_loss`, `cma_sigma` |
| `final_result.json` | Best config, loss, run id |
| `evaluated_pool.json` | Every candidate: `generation`, `objectives`, `config`, `cost_usd` |
| `pareto_front.json` | Non-dominated set (multi-objective runs) |
| `optimization_meta.json` | `population_size`, `generations`, `total_function_evals`, `pareto_size` |

**Plotting** — `sensor_opt/plotting/convergence.py` and `sensor_opt/plotting/paper_figures.py` build **static SVG** strings (NumPy + stdlib) for notebooks: convergence, Pareto 2D scatter (marker size = sensor count, color = cost tier), σ vs generation, slot heatmap, parameter histograms, metric correlation, detection-latency CDF (from data you pass in), baseline bars, hypervolume vs budget, and best-loss vs evaluation count. Import `sensor_opt.plotting` or `sensor_opt.plotting.paper_figures` in Jupyter/Colab.

## Project Structure

```
sensor_placement_opt/
├── notebooks/          # Colab walkthroughs (Isaac Lab + mock HTTP)
├── scripts/            # isaaclab_sensor_bridge.py (HTTP JSON + Isaac)
├── sensor_opt/
│   ├── config/         # apply_sensor_catalog (YAML catalog → sensor_models)
│   ├── encoding/       # Encode/decode sensor configs ↔ float vectors
│   ├── loss/           # L = α·collision + β·blind_spot + γ·cost (modes in YAML)
│   ├── cma/            # CMA-ES outer loop wrapper + Pareto
│   ├── inner_loop/     # Isaac Sim evaluator, bridge client, mock, prism path layout
│   ├── logging/        # CSV, JSON artifacts, MLflow
│   └── plotting/       # SVG convergence + paper figures (no matplotlib)
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

**Loss modes (Isaac / research):** `configs/default.yaml` uses `loss.mode: default` (collision + blind spot + cost). For obstacle-based evaluation with \( \alpha\cdot p95(t_{det}) + \beta\cdot\text{collision} \) (and related fields in `EvalMetrics`), use `configs/obstacle_isaaclab.yaml` (`loss.mode: obstacle_latency`). For the legacy slot layout without the six-site prism, use `configs/obstacle_isaaclab_corridor.yaml` (omit `fixed_mount_order` or set it `false` in a forked YAML).

**`fixed_mount_order` (YAML):** when `true`, the \(i\)-th float block in the CMA-ES vector is decoded with **`mounting_slots[i]`** as the mount name (sensor **type** is still chosen per block). Pairs with six **`mounting_slots`** and a **`sensor_budget`** whose `max_count` values sum to six for a pure per-site assignment search.