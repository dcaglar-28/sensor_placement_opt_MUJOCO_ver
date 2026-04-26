"""
Colab / notebook figures: σ vs best loss, slot coverage, 2D placement (matplotlib).

Preserves CSV column names from `generations.csv` and loss term slot naming.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def plot_sigma_vs_best_loss(
    csv_path: PathLike,
    *,
    title: str | None = None,
    figsize: Tuple[float, float] = (10.0, 4.0),
) -> Any:
    import matplotlib.pyplot as plt

    from sensor_opt.plotting.convergence import load_generations_csv

    p = Path(csv_path)
    gen, cols = load_generations_csv(p)
    o = np.argsort(gen)
    g = gen[o]
    best = cols["best_loss"][o]
    fig, axl = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    c = "tab:blue"
    axl.plot(g, best, "o-", color=c, label="best loss", markersize=3, linewidth=1.5)
    axl.set_xlabel("generation")
    axl.set_ylabel("best loss (total)", color=c)
    axl.tick_params(axis="y", labelcolor=c)
    if "cma_sigma" in cols:
        axr = axl.twinx()
        axr.plot(g, cols["cma_sigma"][o], color="k", alpha=0.6, linewidth=1.2, label="CMA σ")
        axr.set_ylabel("CMA step size σ", color="k")
        axr.tick_params(axis="y", labelcolor="k")
    fig.suptitle(title or f"σ vs loss ({p.parent.name})", fontsize=12)
    return fig


def slot_coverage_heatmap(
    mount_names: Sequence[str],
    per_slot_covered: Sequence[float],
    *,
    title: str = "Per-mount coverage (0=uncovered, 1=covered)",
    figsize: Tuple[float, float] = (10.0, 2.2),
) -> Any:
    import matplotlib.pyplot as plt

    m = np.asarray(per_slot_covered, dtype=np.float64).reshape(1, -1)
    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    im = ax.imshow(m, aspect="auto", cmap="Greens", vmin=0, vmax=1)
    ax.set_yticks([0])
    ax.set_yticklabels(["covered?"])
    ax.set_xticks(np.arange(len(mount_names)))
    ax.set_xticklabels(list(mount_names), rotation=30, ha="right")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.04)
    return fig


def sensor_placement_2d(
    slot_xy: List[Tuple[str, float, float]],
    title: str = "Top-down (offset space)",
    figsize: Tuple[float, float] = (6.0, 5.0),
) -> Any:
    """
    Scatter: each mount at (x_offset, y_offset) with label = sensor type or slot.
    `slot_xy` = [(name, x, y), ...]
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)
    palette = "tab10"
    for i, (name, x, y) in enumerate(slot_xy):
        ax.scatter(
            [x], [y], s=120, label=name[:20], c=[plt.get_cmap(palette)(i % 10)],
        )
        ax.annotate(name[:8], (x, y), textcoords="offset points", xytext=(3, 3), fontsize=7)
    ax.set_xlabel("x_offset (m)")
    ax.set_ylabel("y_offset (m)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7)
    return fig
