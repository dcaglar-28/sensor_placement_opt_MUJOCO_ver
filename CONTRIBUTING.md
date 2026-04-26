# Contributing

## Branch strategy

\```
main          ← stable, always green (CI passes)
  └─ dev      ← integration branch; PRs merge here first
       ├─ phase/0-scaffold      (done)
       ├─ phase/1-inner-loop    (MuJoCo / custom sim)
       ├─ phase/2-rl-inner-loop
       └─ feat/<short-name>     (any standalone feature)
\```

**Never push directly to `main`.** Open a PR from `dev` → `main` when a
phase is complete and all tests pass.

## Starting new work

\```bash
git checkout dev
git pull origin dev
git checkout -b feat/your-feature-name
\```

## Commit message format (Conventional Commits)

\```
<type>(<scope>): <short summary>

types: feat | fix | test | refactor | docs | chore
scope: encoding | loss | cma | inner_loop | logging | config | ci

Examples:
  feat(inner_loop): extend inner-loop evaluator
  fix(encoding): clamp yaw to [-180, 180] before normalising
  test(loss): add edge case for zero-weight gamma
  chore(ci): add GitHub Actions pytest workflow
\```

## Before opening a PR

\```bash
pip install -r requirements.txt
pytest tests/ -v
\```

## Adding a new sensor type

1. Add the type to `SENSOR_TYPE_MAP` in `sensor_opt/encoding/config.py`
2. Add model specs to `configs/default.yaml` under `sensor_models`
3. Add slot coverage + collision weights in `sensor_opt/inner_loop/dummy_evaluator.py`
4. Add unit cost to loss tests in `tests/test_loss.py`