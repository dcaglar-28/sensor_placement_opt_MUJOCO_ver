"""Smoke tests: SVG paper figure helpers (no matplotlib)."""

import json
import tempfile
from pathlib import Path

import pytest

from sensor_opt.plotting.paper_figures import (
    fig01_convergence_multi,
    fig02_pareto_scatter_2d,
    fig03_cma_sigma,
    fig07_correlation_heatmap,
    fig11_sample_efficiency,
)


def _write_generations(path: Path) -> None:
    path.write_text(
        "generation,best_loss,mean_loss,std_loss,cma_sigma\n"
        "0,1.0,1.1,0.1,0.5\n"
        "1,0.8,0.9,0.08,0.45\n"
        "2,0.7,0.85,0.07,0.4\n",
        encoding="utf-8",
    )


def test_fig01_fig03_fig11_svg() -> None:
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "generations.csv"
        _write_generations(p)
        s1 = fig01_convergence_multi([(p, "run_a")])
        s3 = fig03_cma_sigma(p)
        s11 = fig11_sample_efficiency(p)
    assert s1.strip().startswith("<svg")
    assert s3.strip().startswith("<svg")
    assert s11.strip().startswith("<svg")


def test_fig02_pareto() -> None:
    data = [
        {
            "index": 0,
            "objectives": {"collision": 0.1, "blind_spot": 0.2},
            "n_active_sensors": 2,
            "cost_usd": 10.0,
        },
        {
            "index": 1,
            "objectives": {"collision": 0.2, "blind_spot": 0.1},
            "n_active_sensors": 3,
            "cost_usd": 20.0,
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        j = Path(td) / "pareto.json"
        j.write_text(json.dumps(data), encoding="utf-8")
        svg = fig02_pareto_scatter_2d(j)
    assert "<circle" in svg


def test_fig07_correlation() -> None:
    pool = [
        {
            "objectives": {
                "collision": 0.1,
                "blind_spot": 0.2,
                "cost": 1.0,
            },
        },
        {
            "objectives": {
                "collision": 0.15,
                "blind_spot": 0.18,
                "cost": 1.1,
            },
        },
    ]
    with tempfile.TemporaryDirectory() as td:
        j = Path(td) / "pool.json"
        j.write_text(json.dumps(pool), encoding="utf-8")
        svg = fig07_correlation_heatmap(j, keys=("collision", "blind_spot", "cost"))
    assert "0." in svg or "1." in svg or "-0." in svg
