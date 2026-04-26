"""
Matplotlib figures from `generations.csv` (CMA-ES outer loop).

Mirrors the SysID notebook style: per-generation best loss, running best (like
`np.minimum.accumulate` on batch losses), mean ± std band, and a second panel
for weighted loss terms + CMA step size ``σ`` (the optimizer's "input scale").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np

from sensor_opt.plotting.convergence import load_generations_csv

PathLike = Union[str, Path]


def plot_cma_generations_matplotlib(
    csv_path: PathLike,
    *,
    title: Optional[str] = None,
    figsize: Tuple[float, float] = (14.0, 5.0),
) -> Any:
    """
    Build a 1×2 figure: loss convergence | loss terms + CMA σ.

    Requires ``matplotlib`` in the environment.
    """
    import matplotlib.pyplot as plt

    p = Path(csv_path)
    gen, cols = load_generations_csv(p)
    order = np.argsort(gen)
    g = gen[order]
    best = cols["best_loss"][order]
    run_best = np.minimum.accumulate(best)

    t = title or f"CMA-ES run ({p.parent.name})"
    fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

    axes[0].plot(g, best, "o-", label="best in generation", color="steelblue", markersize=4)
    axes[0].plot(
        g,
        run_best,
        "-",
        label="CMA-ES (running best)",
        color="darkorange",
        linewidth=2,
    )
    if "mean_loss" in cols and "std_loss" in cols:
        m = cols["mean_loss"][order]
        s = cols["std_loss"][order]
        axes[0].fill_between(g, m - s, m + s, color="green", alpha=0.2, label="mean ± std")
    axes[0].set_xlabel("generation")
    axes[0].set_ylabel("total loss")
    axes[0].set_title("Convergence")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    if all(k in cols for k in ("best_collision_term", "best_blind_term", "best_cost_term")):
        axes[1].plot(
            g,
            cols["best_collision_term"][order],
            "-",
            label="term: collision slot",
            color="C0",
        )
        axes[1].plot(
            g,
            cols["best_blind_term"][order],
            "-",
            label="term: blind / latency / speed slot",
            color="C1",
        )
        axes[1].plot(
            g,
            cols["best_cost_term"][order],
            "-",
            label="term: cost (γ·…)",
            color="C2",
        )
    if "cma_sigma" in cols:
        axr = axes[1].twinx()
        axr.plot(g, cols["cma_sigma"][order], "k--", alpha=0.55, linewidth=1.2, label="CMA σ")
        axr.set_ylabel("CMA step size σ", color="k")
        h1, l1 = axes[1].get_legend_handles_labels()
        h2, l2 = axr.get_legend_handles_labels()
        axes[1].legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    else:
        axes[1].legend(loc="upper right", fontsize=8)
    axes[1].set_xlabel("generation")
    axes[1].set_ylabel("weighted term value")
    axes[1].set_title("Loss breakdown + search scale")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(t, fontsize=12, y=1.02)
    return fig
