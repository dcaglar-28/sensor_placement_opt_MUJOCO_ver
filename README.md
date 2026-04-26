# Sensor Placement Optimization
**Bi-Level CMA-ES Search for Autonomous Robot Sensor Configurations**

## Architecture

```
Outer Loop (CMA-ES)          Inner Loop (MuJoCo / Mock / custom sim)
─────────────────────        ──────────────────────────────
encode config → σ-vector  →  load env + attach sensors
mutate population         ←  run K=15 episodes
evaluate L = α·col +         return {collision_rate,
         β·blind +                    blind_spot_fraction,
         γ·cost                       sensor_cost}
select + update CMA
log to CSV / MLflow
```

## Quick Start (CPU-only)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m sensor_opt.run_experiment --config configs/default.yaml --dummy
```

Notes:
- `--dummy` is an **alias** for the built-in fast mock inner loop (analytic-style metrics; same as the “mock” branch in `configs/default.yaml`).
- Physics rollouts: set `inner_loop.mode: mujoco` and use e.g. `configs/mujoco.yaml` (see `sensor_opt/inner_loop/mujoco_env_manager.py`).
- To run that mock from YAML, set `inner_loop.mode` to the mock option in `configs/default.yaml` (not `mujoco`).

Tests (after `pip install -r requirements.txt`):

```bash
python -m pytest tests/ -q
```

## Google Colab (optional)

Walkthrough: `notebooks/sensor_placement_opt_colab.ipynb` (MuJoCo on CPU). The first code cell `pip install`s the same **`requirements.txt`** the repo uses (JAX, scikit-learn, MLflow, PyTorch, Matplotlib, pandas, rich, etc.). The notebook also **embeds** the `sensor_opt` tree and smoke YAML; regenerate the notebook with `python3 scripts/emit_colab_sensor_placement_notebook.py` after code changes. No separate clone is required on Colab. For a local checkout, use `pip install -r requirements.txt`, then e.g. `configs/mujoco.yaml` and `python -m sensor_opt.run_experiment` as usual.

**Obstacle / latency research:** use the obstacle+latency example YAMLs in `configs/` (names begin with `obstacle_`…) with `loss.mode: obstacle_latency` and an evaluator that fills `t_det_s_p95` and related `EvalMetrics` fields.

**Paper figures (Colab notebook)** — After an optimization run, cells can render **SVG** plots via `IPython.display.SVG`: convergence, Pareto scatter, CMA-ES **σ** vs generation, etc. Point paths at the run directory (parent of `generations.csv`).

## Sensor catalog (YAML)

Configs can list a **`sensor_catalog`** (per-type prim paths / FOV / cost metadata) and **`sensor_choices` / slot** definitions. At load time, `apply_sensor_catalog()` in `sensor_opt/config/catalog.py` materializes the merged **`sensor_models`** dict used by the encoder and evaluators. This keeps one catalog per hardware family while reusing the same optimization pipeline.

Per-type **inventory** goes in **`sensor_budget`**: `usermax` is how many units of that sensor you *can* install (copied to `max_count` for the encoder if you omit `max_count`). The search can use **any** count from 0 up to that cap; unused capacity stays `disabled` in the encoded layout, so you do not have to use every unit. Optional `min_count` is a floor (e.g. at least one camera). You may set `usermax` and `max_count` to the same integer, or only one of them. Machine hardware fields such as `gpu_cores`, `unified_memory_gb`, and `memory_bandwidth_gbps` are validated only when `inner_loop.mode` selects the external-sim path in `configs/default.yaml`; mock runs do not need them.

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
├── notebooks/          # Colab walkthrough (MuJoCo)
├── sensor_opt/
│   ├── config/         # apply_sensor_catalog (YAML catalog → sensor_models)
│   ├── encoding/       # Encode/decode sensor configs ↔ float vectors
│   ├── loss/           # L = α·collision + β·blind_spot + γ·cost (modes in YAML)
│   ├── cma/            # CMA-ES outer loop wrapper + Pareto
│   ├── inner_loop/     # MuJoCo / mock / sim evaluators, prism path helpers
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
| 1 — MuJoCo inner loop | CPU |
| 2 — Full RL inner loop | Phase 1 + PPO agent |

**Loss modes:** `configs/default.yaml` uses `loss.mode: default` (collision + blind spot + cost). For obstacle-based evaluation with \( \alpha\cdot p95(t_{det}) + \beta\cdot\text{collision} \) (and related fields in `EvalMetrics`), use the `obstacle_*.yaml` examples in `configs/` with `loss.mode: obstacle_latency` (prism path vs corridor differ by layout flags such as `fixed_mount_order`).

**`fixed_mount_order` (YAML):** when `true`, the \(i\)-th float block in the CMA-ES vector is decoded with **`mounting_slots[i]`** as the mount name (sensor **type** is still chosen per block). Pairs with six **`mounting_slots`** and a **`sensor_budget`** whose `max_count` values sum to six for a pure per-site assignment search.
