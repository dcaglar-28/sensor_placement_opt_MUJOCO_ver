# Sensor Placement Optimization
**Bi-Level CMA-ES Search for Autonomous Robot Sensor Configurations**

## Architecture

\```
Outer Loop (CMA-ES)          Inner Loop (Isaac Sim / Mock Isaac)
─────────────────────        ──────────────────────────────
encode config → σ-vector  →  load env + attach sensors
mutate population         ←  run K=15 episodes
evaluate L = α·col +         return {collision_rate,
         β·blind +                    blind_spot_fraction,
         γ·cost                       sensor_cost}
select + update CMA
log to CSV / MLflow
\```

## Quick Start (CPU-only, no Isaac Sim needed)

\```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m sensor_opt.run_experiment --config configs/default.yaml --dummy
\```

Notes:
- `--dummy` is an **alias** for the built-in `mock_isaac` evaluator (no Isaac Sim required).
- To run explicitly via config, set `inner_loop.mode: mock_isaac`.

## Project Structure

\```
sensor_placement_opt/
├── sensor_opt/
│   ├── encoding/       # Encode/decode sensor configs ↔ float vectors
│   ├── loss/           # L = α·collision + β·blind_spot + γ·cost
│   ├── cma/            # CMA-ES outer loop wrapper
│   ├── inner_loop/     # Isaac Sim evaluator + mock (CPU) evaluator
│   └── logging/        # CSV + MLflow experiment tracking
├── configs/            # YAML experiment configs
├── tests/              # Unit tests (pytest)
└── results/            # Auto-created run outputs
\```

## Phases

| Phase | Status | Requires |
|-------|--------|----------|
| 0 — Scaffold + mock eval | CPU-only | Python 3.10+, cma |
| 1 — Isaac Sim integration | Isaac Sim 4.x, GPU |
| 2 — Full RL inner loop | Phase 1 + PPO agent |