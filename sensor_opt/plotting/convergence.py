"""
Convergence visualizations from CMA-ES `generations.csv` logs.

Renders a **static SVG** (NumPy + stdlib only) so notebooks can use
`IPython.display.SVG` in Colab / Jupyter without matplotlib.
"""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


def load_generations_csv(path: PathLike) -> Tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Read a `generations.csv` and return (generation_index, columns).

    `columns` maps header name -> float array (same length).
    """
    p = Path(path)
    with p.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Empty or invalid CSV: {p}")
        rows: List[dict[str, str]] = list(reader)

    if not rows:
        raise ValueError(f"No data rows in {p}")

    numeric_keys = [
        "generation",
        "best_loss",
        "mean_loss",
        "std_loss",
        "cma_sigma",
    ]
    out: dict[str, np.ndarray] = {}
    for k in numeric_keys:
        if k in rows[0]:
            out[k] = np.array([float(r[k]) for r in rows], dtype=np.float64)

    gen = out.get("generation")
    if gen is None:
        raise KeyError("Expected column 'generation' in generations.csv")

    return gen, out


def best_loss_so_far(best_per_gen: Sequence[float]) -> np.ndarray:
    """Cumulative minimum (standard 'best-so-far' for minimization)."""
    x = np.asarray(best_per_gen, dtype=np.float64)
    return np.minimum.accumulate(x)


def _polyline_points(xs: np.ndarray, ys: np.ndarray) -> str:
    parts = [f"{float(x):.4f},{float(y):.4f}" for x, y in zip(xs, ys)]
    return " ".join(parts)


def _scale_x(
    g: np.ndarray, g0: float, g1: float, inner_left: float, inner_w: float
) -> np.ndarray:
    span = max(g1 - g0, 1e-9)
    return inner_left + (g - g0) / span * inner_w


def _scale_y(
    y: np.ndarray, y0: float, y1: float, inner_top: float, inner_h: float
) -> np.ndarray:
    span = max(y1 - y0, 1e-9)
    return inner_top + (1.0 - (y - y0) / span) * inner_h


def plot_convergence_arrays(
    generation: np.ndarray,
    best_loss: np.ndarray,
    mean_loss: Optional[np.ndarray] = None,
    std_loss: Optional[np.ndarray] = None,
    *,
    title: str = "Optimizer convergence",
    ylabel: str = "loss",
    width: int = 720,
    height: int = 400,
) -> str:
    """
    Build an SVG string: per-generation best, best-so-far, optional mean ± std band.

    Returns XML suitable for `IPython.display.SVG` or saving as `.svg`.
    """
    order = np.argsort(generation)
    g = generation[order]
    b = best_loss[order]
    sofar = best_loss_so_far(b)

    y_cands = [b, sofar]
    if mean_loss is not None and std_loss is not None:
        m = mean_loss[order]
        s = std_loss[order]
        y_cands.extend([m - s, m + s, m])
    y0 = float(np.min([np.min(x) for x in y_cands]))
    y1 = float(np.max([np.max(x) for x in y_cands]))
    pad = 0.05 * (y1 - y0 + 1e-9)
    y0 -= pad
    y1 += pad

    g0, g1 = float(g[0]), float(g[-1])
    if g1 - g0 < 1e-9:
        g0 -= 0.5
        g1 += 0.5

    margin_l, margin_r, margin_t, margin_b = 58.0, 24.0, 44.0, 48.0
    inner_l = margin_l
    inner_t = margin_t
    inner_w = width - margin_l - margin_r
    inner_h = height - margin_t - margin_b

    xg = _scale_x(g, g0, g1, inner_l, inner_w)
    xb = _scale_y(b, y0, y1, inner_t, inner_h)
    xs = _scale_y(sofar, y0, y1, inner_t, inner_h)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2!s}" y="28" text-anchor="middle" font-size="14" font-family="system-ui, sans-serif">'
        f"{html.escape(title)}</text>",
    ]

    # Axes
    x1 = inner_l + inner_w
    yb_axis = inner_t + inner_h
    parts.append(
        f'<line x1="{inner_l!s}" y1="{yb_axis!s}" x2="{x1!s}" y2="{yb_axis!s}" stroke="#333" stroke-width="1"/>'
    )
    parts.append(
        f'<line x1="{inner_l!s}" y1="{inner_t!s}" x2="{inner_l!s}" y2="{yb_axis!s}" stroke="#333" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="16" y="{height/2!s}" transform="rotate(-90 16 {height/2!s})" text-anchor="middle" '
        f'font-size="12" font-family="system-ui, sans-serif">{html.escape(ylabel)}</text>'
    )
    parts.append(
        f'<text x="{(inner_l + x1)/2!s}" y="{height - 12!s}" text-anchor="middle" font-size="12" '
        f'font-family="system-ui, sans-serif">generation</text>'
    )

    # Optional mean ± std
    if mean_loss is not None and std_loss is not None and len(order):
        m = mean_loss[order]
        s = std_loss[order]
        y_hi = m + s
        y_lo = m - s
        y_hi_s = _scale_y(y_hi, y0, y1, inner_t, inner_h)
        y_lo_s = _scale_y(y_lo, y0, y1, inner_t, inner_h)
        pts_up = " ".join(f"{float(x)},{float(y)}" for x, y in zip(xg, y_hi_s))
        pts_lo_rev = " ".join(f"{float(x)},{float(y)}" for x, y in zip(xg[::-1], y_lo_s[::-1]))
        parts.append(
            f'<polygon points="{pts_up} {pts_lo_rev}" fill="#98df8a" fill-opacity="0.35" stroke="none"/>'
        )
        y_m = _scale_y(m, y0, y1, inner_t, inner_h)
        parts.append(
            f'<polyline points="{_polyline_points(xg, y_m)}" fill="none" stroke="#2ca02c" stroke-width="1.2" stroke-dasharray="4 3"/>'
        )

    # Best in generation, best-so-far
    parts.append(
        f'<polyline points="{_polyline_points(xg, xb)}" fill="none" stroke="#1f77b4" stroke-width="1.4"/>'
    )
    for i in range(len(xg)):
        parts.append(
            f'<circle cx="{float(xg[i]):.2f}" cy="{float(xb[i]):.2f}" r="2.2" fill="#1f77b4"/>'
        )
    parts.append(
        f'<polyline points="{_polyline_points(xg, xs)}" fill="none" stroke="#000" stroke-width="1.6"/>'
    )

    # Legend
    ly = margin_t + 6
    lx = inner_l + inner_w - 200
    parts += [
        f'<rect x="{lx-4}" y="{ly-2}" width="200" height="66" fill="white" fill-opacity="0.85" stroke="#ddd"/>',
        f'<line x1="{lx}" y1="{ly+6}" x2="{lx+20}" y2="{ly+6}" stroke="#1f77b4" stroke-width="1.4"/>',
        f'<text x="{lx+26}" y="{ly+10}" font-size="10" font-family="system-ui, sans-serif">best in gen</text>',
        f'<line x1="{lx}" y1="{ly+24}" x2="{lx+20}" y2="{ly+24}" stroke="#000" stroke-width="1.4"/>',
        f'<text x="{lx+26}" y="{ly+28}" font-size="10" font-family="system-ui, sans-serif">best-so-far</text>',
    ]
    if mean_loss is not None and std_loss is not None:
        parts += [
            f'<line x1="{lx}" y1="{ly+44}" x2="{lx+20}" y2="{ly+44}" stroke="#2ca02c" stroke-width="1.2" stroke-dasharray="4 3"/>',
            f'<text x="{lx+26}" y="{ly+48}" font-size="10" font-family="system-ui, sans-serif">mean ± std</text>',
        ]

    parts.append("</svg>")
    return "\n".join(parts)


def plot_convergence_from_csv(
    csv_path: PathLike,
    *,
    save_path: Optional[PathLike] = None,
    title: Optional[str] = None,
) -> str:
    """
    Load `generations.csv` and return an SVG string.

    If `save_path` is set, writes UTF-8 SVG (e.g. for a paper or figure supplement).
    """
    gen, cols = load_generations_csv(csv_path)
    t = title or f"Convergence ({Path(csv_path).name})"
    svg = plot_convergence_arrays(
        gen,
        cols["best_loss"],
        mean_loss=cols.get("mean_loss"),
        std_loss=cols.get("std_loss"),
        title=t,
    )
    if save_path is not None:
        Path(save_path).write_text(svg, encoding="utf-8")
    return svg
