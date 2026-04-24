from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from sensor_opt.plotting.convergence import best_loss_so_far, load_generations_csv


def test_best_loss_so_far_monotonic():
    b = [1.0, 0.8, 0.9, 0.3]
    sf = best_loss_so_far(b)
    assert list(sf) == [1.0, 0.8, 0.8, 0.3]


def test_load_generations_csv_minimal(tmp_path: Path):
    p = tmp_path / "generations.csv"
    rows = [
        {
            "run_id": "x",
            "experiment_name": "e",
            "generation": 1,
            "elapsed_sec": 0.0,
            "best_loss": 0.5,
            "mean_loss": 0.6,
            "std_loss": 0.1,
            "best_collision_term": 0.0,
            "best_blind_term": 0.0,
            "best_cost_term": 0.0,
            "best_cost_usd": 0.0,
            "best_n_active": 1,
            "best_config_summary": "x",
            "population_size": 4,
            "cma_sigma": 0.3,
            "mean_eval_time_sec": 0.0,
            "dominant_fidelity": "single",
        }
    ]
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    gen, cols = load_generations_csv(p)
    assert gen.shape == (1,)
    assert cols["best_loss"][0] == 0.5


def test_plot_convergence_svg_smoke():
    from sensor_opt.plotting.convergence import plot_convergence_arrays

    g = np.array([1, 2, 3], dtype=np.float64)
    b = np.array([0.5, 0.4, 0.6], dtype=np.float64)
    m = np.array([0.55, 0.5, 0.55], dtype=np.float64)
    s = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    svg = plot_convergence_arrays(g, b, mean_loss=m, std_loss=s, title="t")
    assert "<svg" in svg
    assert "best-so-far" in svg
