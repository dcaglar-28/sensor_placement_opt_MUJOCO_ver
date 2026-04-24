"""
Paper-oriented figures (static SVG, NumPy + stdlib) for sensor placement optimization.

Priority (space-constrained papers): Pareto front, convergence, best layout, baseline comparison.
See module docstring sections for data sources and figure IDs 1–11.

Data produced by CMA-ES runs (after `run_outer_loop` / Colab optimizer cell):
  - results/<run_id>/generations.csv
  - results/<run_id>/evaluated_pool.json   (all candidates)
  - results/<run_id>/pareto_front.json     (non-dominated set, with cost_usd, configs)
  - results/<run_id>/optimization_meta.json
"""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from sensor_opt.plotting.convergence import best_loss_so_far, load_generations_csv

PathLike = Union[str, Path]

_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")


# --- 1) Multi-run convergence: best_loss + mean±std per run, overlaid ---


def fig01_convergence_multi(
    run_csvs: Sequence[Tuple[PathLike, str]],
    *,
    title: str = "Convergence (multi-objective / multi-config)",
    width: int = 780,
    height: int = 420,
) -> str:
    """
    Overlay `best_loss` vs `generation` for several `generations.csv` files.

    Also draws a light mean±std band per run (uses `mean_loss` and `std_loss` columns).
    `run_csvs` is a list of (path, label).
    """
    if not run_csvs:
        raise ValueError("run_csvs is empty")
    series: List[dict] = []
    y_min, y_max = 1e9, -1e9
    g_min, g_max = 1e9, -1e9
    for path, _lab in run_csvs:
        gen, cols = load_generations_csv(path)
        g = gen[np.argsort(gen)]
        b = cols["best_loss"][np.argsort(gen)]
        m = cols.get("mean_loss")
        s = cols.get("std_loss")
        sofar = best_loss_so_far(b)
        y_min = min(y_min, float(np.min(b)), float(np.min(sofar)))
        y_max = max(y_max, float(np.max(b)), float(np.max(sofar)))
        g_min = min(g_min, float(g[0]), float(g[-1]))
        g_max = max(g_max, float(g[0]), float(g[-1]))
        if m is not None and s is not None:
            m1 = m[np.argsort(gen)]
            s1 = s[np.argsort(gen)]
            y_min = min(y_min, float(np.min(m1 - s1)))
            y_max = max(y_max, float(np.max(m1 + s1)))
        series.append(
            {
                "g": g,
                "b": b,
                "sofar": sofar,
                "mean": m[np.argsort(gen)] if m is not None else None,
                "std": s[np.argsort(gen)] if s is not None else None,
            }
        )
    pad_y = 0.05 * (y_max - y_min + 1e-9)
    y0, y1 = y_min - pad_y, y_max + pad_y
    if g_max - g_min < 1e-9:
        g_min -= 0.5
        g_max += 0.5

    margin_l, margin_r, margin_t, margin_b = 58.0, 120.0, 44.0, 48.0
    inner_l = margin_l
    inner_t = margin_t
    inner_w = width - margin_l - margin_r
    inner_h = height - margin_t - margin_b
    w, h = width, height

    def _sx(gg: np.ndarray) -> np.ndarray:
        return inner_l + (gg - g_min) / max(g_max - g_min, 1e-9) * inner_w

    def _sy(yy: np.ndarray) -> np.ndarray:
        return inner_t + (1.0 - (yy - y0) / max(y1 - y0, 1e-9)) * inner_h

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="28" text-anchor="middle" font-size="14" font-family="system-ui, sans-serif">'
        f"{html.escape(title)}</text>",
    ]
    yb = inner_t + inner_h
    parts.append(
        f'<line x1="{inner_l}" y1="{yb}" x2="{inner_l + inner_w}" y2="{yb}" stroke="#333"/>'
    )
    parts.append(
        f'<line x1="{inner_l}" y1="{inner_t}" x2="{inner_l}" y2="{yb}" stroke="#333"/>'
    )
    parts.append(
        f'<text x="14" y="{h/2}" transform="rotate(-90 14 {h/2})" text-anchor="middle" font-size="12">loss</text>'
    )
    parts.append(
        f'<text x="{(2*inner_l + inner_w)/2}" y="{h-12}" text-anchor="middle" font-size="12">generation</text>'
    )

    for i, ((path, lab), S) in enumerate(zip(run_csvs, series)):
        col = _COLORS[i % len(_COLORS)]
        g, b, m, s = S["g"], S["b"], S["mean"], S["std"]
        xg = _sx(g)
        if m is not None and s is not None:
            lo, hi = m - s, m + s
            pts1 = " ".join(
                f"{float(x)},{float(y)}"
                for x, y in zip(xg, _sy(hi))
            )
            pts0 = " ".join(
                f"{float(x)},{float(y)}"
                for x, y in zip(xg[::-1], _sy(lo[::-1]))
            )
            parts.append(
                f'<polygon points="{pts1} {pts0}" fill="{col}" fill-opacity="0.12" stroke="none"/>'
            )
        yb1 = _sy(b)
        yso = _sy(S["sofar"])
        pts_b = " ".join(f"{float(x)},{float(y)}" for x, y in zip(xg, yb1))
        pts_s = " ".join(f"{float(x)},{float(y)}" for x, y in zip(xg, yso))
        parts.append(
            f'<polyline points="{pts_b}" fill="none" stroke="{col}" stroke-width="1.2" stroke-opacity="0.9"/>'
        )
        parts.append(
            f'<polyline points="{pts_s}" fill="none" stroke="{col}" stroke-width="1.5" stroke-dasharray="5 3"/>'
        )
        lx = width - margin_r + 8
        ly = margin_t + 20 + i * 36
        parts += [
            f'<line x1="{lx-4}" y1="{ly+4}" x2="{lx+16}" y2="{ly+4}" stroke="{col}" stroke-width="1.4"/>',
            f'<line x1="{lx-4}" y1="{ly+18}" x2="{lx+16}" y2="{ly+18}" stroke="{col}" stroke-width="1.4" stroke-dasharray="5 3"/>',
            f'<text x="{lx+20}" y="{ly+8}" font-size="9" font-family="system-ui, sans-serif">{html.escape(lab)[:32]}</text>',
        ]

    parts.append(
        f'<text x="{width - margin_r + 4}" y="{margin_t + 8}" font-size="8" fill="#666">solid=best in gen, dash=best-so-far</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)


# --- 2) Pareto / objective scatter (2D) ---


def _cost_tier(costs: np.ndarray) -> np.ndarray:
    if costs.size == 0:
        return np.array([], dtype=int)
    q1, q2 = np.quantile(costs, [0.33, 0.66])
    t = np.zeros(len(costs), dtype=int)
    t[costs <= q1] = 0
    t[(costs > q1) & (costs <= q2)] = 1
    t[costs > q2] = 2
    return t


def fig02_pareto_scatter_2d(
    pareto_json: PathLike,
    *,
    x_key: str = "collision",
    y_key: str = "blind_spot",
    title: str = "Pareto-style objective trade-offs",
    width: int = 520,
    height: int = 480,
) -> str:
    """
    2D scatter: objectives[x_key] vs objectives[y_key].
    Marker radius scales with `n_active_sensors`; fill color = cost tier (tertile of cost_usd).
    """
    data = json.loads(Path(pareto_json).read_text(encoding="utf-8"))
    xs: List[float] = []
    ys: List[float] = []
    sizes: List[int] = []
    costs: List[float] = []
    for row in data:
        obj = row.get("objectives") or {}
        if x_key not in obj or y_key not in obj:
            continue
        xs.append(float(obj[x_key]))
        ys.append(float(obj[y_key]))
        sizes.append(int(row.get("n_active_sensors", 1)))
        costs.append(float(row.get("cost_usd", 0.0)))
    if not xs:
        raise ValueError("No points with required objective keys; check loss.mode and JSON.")
    x_a, y_a = np.array(xs), np.array(ys)
    c_a = np.array(costs, dtype=np.float64)
    tier = _cost_tier(c_a)
    tier_cols = ("#2ca02c", "#ffbb78", "#d62728")
    rmin, rmax = 3.0, 12.0
    smin = max(1, int(np.min(sizes)))
    smax = int(np.max(sizes))
    smax = max(smax, smin + 1)

    x0, x1 = float(np.min(x_a)), float(np.max(x_a))
    y0, y1 = float(np.min(y_a)), float(np.max(y_a))
    padx = 0.06 * (x1 - x0 + 1e-9)
    pady = 0.06 * (y1 - y0 + 1e-9)
    x0, x1, y0, y1 = x0 - padx, x1 + padx, y0 - pady, y1 + pady

    ml, mr, mt, mb = 52.0, 180.0, 40.0, 44.0
    iw, ih = width - ml - mr, height - mt - mb
    w, h = width, height

    def px(xx: float) -> float:
        return ml + (xx - x0) / max(x1 - x0, 1e-9) * iw

    def py(yy: float) -> float:
        return mt + (1.0 - (yy - y0) / max(y1 - y0, 1e-9)) * ih

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="24" text-anchor="middle" font-size="14">{html.escape(title)}</text>',
    ]
    yb = mt + ih
    parts += [
        f'<line x1="{ml}" y1="{yb}" x2="{ml+iw}" y2="{yb}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{yb}" stroke="#333"/>',
        f'<text x="{w/2}" y="{h-10}" text-anchor="middle" font-size="11">{html.escape(x_key)}</text>',
        f'<text x="16" y="{h/2}" transform="rotate(-90 16 {h/2})" text-anchor="middle" font-size="11">{html.escape(y_key)}</text>',
    ]
    for i in range(len(xs)):
        rad = rmin + (sizes[i] - smin) / max(smax - smin, 1) * (rmax - rmin)
        tc = tier_cols[int(tier[i])]
        parts.append(
            f'<circle cx="{px(xs[i]):.2f}" cy="{py(ys[i]):.2f}" r="{rad:.1f}" fill="{tc}" fill-opacity="0.75" stroke="#222" stroke-width="0.4"/>'
        )
    # Legend
    lx = width - mr + 10
    parts += [
        f'<text x="{lx}" y="{mt+20}" font-size="10" font-weight="600">Cost tier (tertile)</text>',
        f'<rect x="{lx}" y="{mt+28}" width="10" height="10" fill="{tier_cols[0]}"/>',
        f'<text x="{lx+16}" y="{mt+37}" font-size="9">low</text>',
        f'<rect x="{lx}" y="{mt+44}" width="10" height="10" fill="{tier_cols[1]}"/>',
        f'<text x="{lx+16}" y="{mt+53}" font-size="9">mid</text>',
        f'<rect x="{lx}" y="{mt+60}" width="10" height="10" fill="{tier_cols[2]}"/>',
        f'<text x="{lx+16}" y="{mt+69}" font-size="9">high</text>',
        f'<text x="{lx}" y="{mt+88}" font-size="9">marker size ∝ n_sensors</text>',
    ]
    parts.append("</svg>")
    return "\n".join(parts)


# --- 3) CMA-ES step size σ vs generation ---


def fig03_cma_sigma(generations_csv: PathLike, *, width: int = 640, height: int = 340) -> str:
    gen, cols = load_generations_csv(generations_csv)
    sig = cols["cma_sigma"]
    order = np.argsort(gen)
    g = gen[order]
    s = sig[order]
    w, h = width, height
    ml, mr, mt, mb = 50.0, 24.0, 40.0, 44.0
    iw, ih = w - ml - mr, h - mt - mb
    g0, g1 = float(g[0]), float(g[-1])
    s0, s1 = float(np.min(s)), float(np.max(s))
    s1 += 0.05 * (s1 - s0 + 1e-9)
    s0 = max(0.0, s0 - 0.05 * (s1 - s0))

    def px(gg: float) -> float:
        return ml + (gg - g0) / max(g1 - g0, 1e-9) * iw

    def ps(ss: float) -> float:
        return mt + (1.0 - (ss - s0) / max(s1 - s0, 1e-9)) * ih

    pts = " ".join(f"{px(float(gg)):.2f},{ps(float(ss)):.2f}" for gg, ss in zip(g, s))
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="22" text-anchor="middle" font-size="14">CMA-ES step size σ</text>',
        f'<line x1="{ml}" y1="{mt+ih}" x2="{ml+iw}" y2="{mt+ih}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ih}" stroke="#333"/>',
        f'<polyline points="{pts}" fill="none" stroke="#000" stroke-width="1.4"/>',
        f'<text x="{w/2}" y="{h-8}" text-anchor="middle" font-size="11">generation</text>',
        f'<text x="12" y="{h/2}" transform="rotate(-90 12 {h/2})" text-anchor="middle" font-size="11">σ</text>',
        "</svg>",
    ]
    return "\n".join(body)


# --- 7) Metric correlation (Pearson) heatmap ---


def fig07_correlation_heatmap(
    evaluated_pool_json: PathLike,
    keys: Optional[Sequence[str]] = None,
    *,
    width: int = 500,
    height: int = 500,
) -> str:
    """
    Pearson correlation of scalar objective / metric fields across all evaluated candidates.
    """
    pool = json.loads(Path(evaluated_pool_json).read_text(encoding="utf-8"))
    if keys is None:
        keys = ("collision", "blind_spot", "cost", "t_det_s_p95", "safety_success", "hardware")
    rows: Dict[str, np.ndarray] = {}
    n = len(pool)
    for k in keys:
        v = np.zeros(n, dtype=np.float64)
        ok = True
        for i, row in enumerate(pool):
            o = (row.get("objectives") or {})
            if k not in o:
                ok = False
                break
            v[i] = float(o[k])
        if ok and n > 1:
            rows[k] = v
    if len(rows) < 2:
        raise ValueError("Not enough overlapping metric keys for correlation (check objectives).")
    names = list(rows.keys())
    m = len(names)
    M = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(m):
            a, b = rows[names[i]], rows[names[j]]
            M[i, j] = float(
                np.corrcoef(a, b)[0, 1] if np.std(a) > 1e-12 and np.std(b) > 1e-12 else float("nan")
            )

    cell = min((width - 100) // m, (height - 100) // m, 64)
    ox, oy = 80.0, 40.0
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-size="14">Metric correlation (Pearson)</text>',
    ]
    for i in range(m):
        for j in range(m):
            val = M[i, j]
            t = 0.0 if math.isnan(val) else (val + 1) / 2.0
            col = f"rgb({int(255*(1-t))},{int(200*t)},{int(255*t)})" if not math.isnan(val) else "#eee"
            x = ox + j * cell
            y = oy + i * cell
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell-2}" height="{cell-2}" fill="{col}" stroke="#333" stroke-width="0.3"/>'
            )
            if not math.isnan(val):
                parts.append(
                    f'<text x="{x+cell/2-2}" y="{y+cell/2+4}" text-anchor="middle" font-size="8">{val:.2f}</text>'
                )
        parts.append(
            f'<text x="8" y="{oy + (i+0.5)*cell}" font-size="9" dominant-baseline="middle">{html.escape(names[i][:12])}</text>'
        )
    for j in range(m):
        parts.append(
            f'<text x="{ox + (j+0.5)*cell}" y="{oy - 4}" text-anchor="middle" font-size="8">{html.escape(names[j][:10])}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


# --- 4) Top-down layout (simplified) ---


def fig04_topdown_sensors(
    best_config: dict,
    sensor_models: Optional[dict] = None,
    *,
    title: str = "Best sensor layout (schematic)",
    width: int = 480,
    height: int = 480,
) -> str:
    """
    `best_config` is `final_result.json`['best_config'] or equivalent with `sensors` list.
    Draws a robot body circle and wedge FoV for each **active** non-disabled sensor.
    `sensor_models[sensor_type].horizontal_fov_deg` used when present (default 60°).
    """
    sensors = (best_config or {}).get("sensors") or []
    w, h = width, height
    cx, cy = w / 2, h / 2
    R_body = min(w, h) * 0.12
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="#fafafa"/>',
        f'<text x="{w/2}" y="22" text-anchor="middle" font-size="14">{html.escape(title)}</text>',
        f'<circle cx="{cx}" cy="{cy}" r="{R_body}" fill="#888" fill-opacity="0.35" stroke="#333"/>',
    ]
    sm = sensor_models or {}
    scale = min(w, h) / 5.0
    for s in sensors:
        if s.get("type") == "disabled":
            continue
        yaw = math.radians(float(s.get("yaw_deg", 0.0)))
        fov = 60.0
        t = s.get("type", "")
        if t in sm and isinstance(sm[t], dict):
            fov = float((sm[t].get("horizontal_fov_deg") or fov))
        fov = min(fov, 160.0)
        rng = 0.4 + 0.6 * float(s.get("range_fraction", 1.0))
        L = scale * rng
        a0 = yaw - math.radians(fov / 2)
        a1 = yaw + math.radians(fov / 2)
        x0, y0 = cx + 0.5 * math.cos(a0) * R_body, cy - 0.5 * math.sin(a0) * R_body
        x1, y1 = cx + L * math.cos(a0), cy - L * math.sin(a0)
        x2, y2 = cx + L * math.cos(a1), cy - L * math.sin(a1)
        col = _COLORS[abs(hash(str(t))) % len(_COLORS)]
        points = f"{cx},{cy} {x1},{y1} {x2},{y2}"
        parts.append(
            f'<polygon points="{points}" fill="{col}" fill-opacity="0.25" stroke="{col}" stroke-width="0.6"/>'
        )
        parts.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx + 0.8*L*math.cos(yaw)}" y2="{cy - 0.8*L*math.sin(yaw)}" stroke="{col}" stroke-width="1.2"/>'
        )
        parts.append(
            f'<text x="{cx + L*math.cos(yaw) + 4}" y="{cy - L*math.sin(yaw)}" font-size="9">{html.escape(str(t)[:8])}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


# --- 8) CDF of per-episode values ---


def fig08_cdf(
    values: Sequence[float],
    *,
    label: str = "t_det (s)",
    title: str = "Detection latency CDF (episodes)",
    width: int = 560,
    height: float = 360,
) -> str:
    """Pass per-episode `t_det` samples (e.g. from a custom log); shows empirical CDF."""
    v = np.sort(np.array(list(values), dtype=np.float64))
    n = len(v)
    if n < 1:
        raise ValueError("values is empty")
    p = (np.arange(1, n + 1) - 0.5) / n
    w, h = width, int(height)
    ml, mr, mt, mb = 50.0, 24.0, 40.0, 40.0
    iw, ih = w - ml - mr, h - mt - mb
    x0, x1 = float(v[0]), float(v[-1])
    span = x1 - x0
    x0, x1 = x0 - 0.02 * max(span, 1e-6), x1 + 0.02 * max(span, 1e-6)

    def px(x: float) -> float:
        return ml + (x - x0) / max(x1 - x0, 1e-9) * iw

    def py(pp: float) -> float:
        return mt + (1.0 - pp) * ih

    pts = " ".join(f"{px(float(vi)):.2f},{py(float(pi)):.2f}" for vi, pi in zip(v, p))
    body = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="22" text-anchor="middle" font-size="14">{html.escape(title)}</text>',
        f'<line x1="{ml}" y1="{mt+ih}" x2="{ml+iw}" y2="{mt+ih}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ih}" stroke="#333"/>',
        f'<polyline points="{pts}" fill="none" stroke="#1f55a5" stroke-width="1.6"/>',
        f'<text x="{w/2}" y="{h-8}" text-anchor="middle" font-size="11">{html.escape(label)}</text>',
        f'<text x="10" y="{h/2}" transform="rotate(-90 10 {h/2})" text-anchor="middle" font-size="11">F(x)</text>',
        "</svg>",
    ]
    return "\n".join(body)


# --- 9) Grouped bar chart (baseline vs optimized vs random) ---


def fig09_baseline_bars(
    series: Dict[str, Dict[str, float]],
    *,
    title: str = "Baselines vs optimized",
    width: int = 640,
    height: int = 380,
) -> str:
    """
    `series`: {method_name: {metric_name: value}} e.g. {"hand": {"collision":0.2}, "ours": {"collision":0.05}}.
    """
    methods = list(series.keys())
    if not methods:
        raise ValueError("empty series")
    metrics = list(series[methods[0]].keys())
    n_m, n_k = len(methods), len(metrics)
    w, h = width, height
    group_w = (w - 100) / max(n_k, 1)
    bar_w = group_w * 0.7 / max(n_m, 1)
    ymax = max(
        max(series[m].get(k, 0.0) for m in methods) for k in metrics
    )
    ymax *= 1.1
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="20" text-anchor="middle" font-size="14">{html.escape(title)}</text>',
    ]
    base_y = h - 50
    top_y = 50
    scale_h = base_y - top_y
    for ki, k in enumerate(metrics):
        gx = 60 + ki * (w - 100) / max(n_k, 1)
        for mi, m in enumerate(methods):
            val = float(series[m].get(k, 0.0))
            bh = (val / ymax) * scale_h
            x = gx + mi * bar_w
            y = base_y - bh
            col = _COLORS[mi % len(_COLORS)]
            parts.append(
                f'<rect x="{x}" y="{y}" width="{bar_w*0.9}" height="{bh}" fill="{col}" fill-opacity="0.85" stroke="#333" stroke-width="0.2"/>'
            )
        parts.append(
            f'<text x="{gx + group_w/2}" y="{h-30}" text-anchor="middle" font-size="9">{html.escape(k)[:20]}</text>'
        )
    for mi, m in enumerate(methods):
        parts.append(
            f'<text x="10" y="{50 + mi * 16}" font-size="9" fill="{_COLORS[mi % len(_COLORS)]}">■ {html.escape(m)[:20]}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


# --- 10) Hypervolume vs budget (line) ---


def fig10_hypervolume_vs_budget(
    points: Sequence[Tuple[float, float]],
    *,
    title: str = "Hardware budget vs Pareto hypervolume (proxy)",
    width: int = 600,
    height: int = 360,
) -> str:
    """`points` = (cost_cap_usd, hypervolume) from multiple ablation runs (you compute HV offline)."""
    if not points:
        raise ValueError("empty points")
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    w, h = width, int(height)
    ml, mr, mt, mb = 52.0, 24.0, 40.0, 44.0
    iw, ih = w - ml - mr, h - mt - mb
    x0, x1 = float(np.min(xs)), float(np.max(xs))
    y0, y1 = float(np.min(ys)), float(np.max(ys))
    x0, x1 = x0 - 0.05 * (x1 - x0), x1 + 0.05 * (x1 - x0)
    y0, y1 = y0 - 0.05 * (y1 - y0), y1 + 0.05 * (y1 - y0)

    def px(x: float) -> float:
        return ml + (x - x0) / max(x1 - x0, 1e-9) * iw

    def py(y: float) -> float:
        return mt + (1.0 - (y - y0) / max(y1 - y0, 1e-9)) * ih

    pts = " ".join(
        f"{px(float(x)):.2f},{py(float(y)):.2f}" for x, y in zip(xs, ys)
    )
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="20" text-anchor="middle" font-size="14">{html.escape(title)}</text>',
        f'<line x1="{ml}" y1="{mt+ih}" x2="{ml+iw}" y2="{mt+ih}" stroke="#333"/>',
        f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ih}" stroke="#333"/>',
        f'<polyline points="{pts}" fill="none" stroke="#2ca02c" stroke-width="1.4"/>',
        f'<text x="{w/2}" y="{h-10}" text-anchor="middle" font-size="11">cost cap (USD)</text>',
        f'<text x="10" y="{h/2}" transform="rotate(-90 10 {h/2})" text-anchor="middle" font-size="11">hypervolume</text>',
        "</svg>",
    ]
    return "\n".join(out)


# --- 11) Best loss vs cumulative evaluations (sample efficiency) ---


def fig11_sample_efficiency(
    generations_csv: PathLike,
    *,
    pop_size: Optional[int] = None,
    title: str = "Best loss vs function evaluations (sample efficiency)",
    width: int = 680,
    height: int = 380,
) -> str:
    """
    X = cumulative number of function evaluations (generations × population).
    If `pop_size` is None, it is read from `optimization_meta.json` next to the CSV when present.
    """
    p = Path(generations_csv)
    gen, cols = load_generations_csv(p)
    order = np.argsort(gen)
    g = gen[order]
    b = cols["best_loss"][order]
    sofar = best_loss_so_far(b)
    if pop_size is None:
        meta = p.parent / "optimization_meta.json"
        pop_size = 20
        if meta.is_file():
            pop_size = int(json.loads(meta.read_text()).get("population_size", pop_size) or pop_size)
    cum = g * float(pop_size)
    w, h = width, height
    ml, mr, mt, mb = 54.0, 24.0, 40.0, 44.0
    iw, ih = w - ml - mr, h - mt - mb
    c0, c1 = float(cum[0]), float(cum[-1])
    y0, y1 = float(np.min(sofar)), float(np.max(b))
    y0, y1 = y0 - 0.05 * (y1 - y0), y1 + 0.05 * (y1 - y0)

    def pc(c: float) -> float:
        return ml + (c - c0) / max(c1 - c0, 1e-9) * iw

    def pyv(y: float) -> float:
        return mt + (1.0 - (y - y0) / max(y1 - y0, 1e-9)) * ih

    pts = " ".join(
        f"{pc(float(c)):.2f},{pyv(float(y)):.2f}" for c, y in zip(cum, sofar)
    )
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{w/2}" y="20" text-anchor="middle" font-size="14">{html.escape(title)}</text>',
            f'<line x1="{ml}" y1="{mt+ih}" x2="{ml+iw}" y2="{mt+ih}" stroke="#333"/>',
            f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ih}" stroke="#333"/>',
            f'<polyline points="{pts}" fill="none" stroke="#d62728" stroke-width="1.6"/>',
            f'<text x="{w/2}" y="{h-10}" text-anchor="middle" font-size="11">cumulative evaluations (≈ gen × pop)</text>',
            f'<text x="12" y="{h/2}" transform="rotate(-90 12 {h/2})" text-anchor="middle" font-size="11">best loss (so-far)</text>',
            "</svg>",
        ]
    )


# --- 5) Slot heatmap (JSON matrix you build from several runs) ---


def fig05_slot_heatmap(
    matrix: Dict[str, Dict[str, str]],
    *,
    title: str = "Slot vs hardware goal (sensor type placed)",
    width: int = 640,
    height: int = 400,
) -> str:
    """
    `matrix[slot][goal] = "lidar"` etc.  Goals = columns, slots = rows.
    """
    slots = list(matrix.keys())
    goals = list(matrix[slots[0]].keys()) if slots else []
    n_r, n_c = len(slots), len(goals)
    if n_r == 0 or n_c == 0:
        raise ValueError("empty matrix")
    w, h = width, max(height, 60 + 28 * n_r)
    cell_w = (w - 120) / max(n_c, 1)
    cell_h = 28.0
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="20" text-anchor="middle" font-size="14">{html.escape(title)}</text>',
    ]
    for j, g in enumerate(goals):
        parts.append(
            f'<text x="{100 + (j+0.5)*cell_w}" y="40" text-anchor="middle" font-size="9">{html.escape(g)[:12]}</text>'
        )
    for i, s in enumerate(slots):
        y = 50 + i * cell_h
        parts.append(
            f'<text x="8" y="{y+cell_h/2+4}" font-size="9">{html.escape(s)[:14]}</text>'
        )
        for j, g in enumerate(goals):
            v = (matrix.get(s) or {}).get(g, "—")
            hue = (hash(v) % 360) / 360.0
            fill = f"hsl({int(hue*360)}, 55%, 85%)"
            x = 100 + j * cell_w
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w-2}" height="{cell_h-4}" fill="{fill}" stroke="#999" stroke-width="0.3"/>'
            )
            parts.append(
                f'<text x="{x+cell_w/2}" y="{y+cell_h/2+3}" text-anchor="middle" font-size="8">{html.escape(str(v)[:8])}</text>'
            )
    parts.append("</svg>")
    return "\n".join(parts)


# --- 6) Simplified "violin" = stacked horizontal density stripes per parameter (last generation filter optional) ---


def fig06_param_distributions(
    evaluated_pool_json: PathLike,
    param: str = "yaw_deg",
    last_generation_only: bool = True,
    *,
    width: int = 600,
    height: int = 360,
) -> str:
    """
    Histogram / stripe plot for one pose parameter across evaluated candidates
    (optionally only the max generation — proxy for "final population").
    """
    pool = json.loads(Path(evaluated_pool_json).read_text(encoding="utf-8"))
    if last_generation_only and pool:
        gmax = max(int(r.get("generation", 0)) for r in pool)
        pool = [r for r in pool if int(r.get("generation", 0)) == gmax]
    xs: List[float] = []
    for r in pool:
        for s in (r.get("config") or {}).get("sensors") or []:
            if s.get("type", "disabled") == "disabled":
                continue
            if param in s:
                xs.append(float(s[param]))
    if len(xs) < 2:
        raise ValueError("Not enough samples for distribution plot.")
    v = np.array(xs, dtype=np.float64)
    lo, hi = float(np.min(v)), float(np.max(v))
    lo, hi = lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo)
    bins = min(20, max(5, int(len(v) ** 0.5)))
    hist, edges = np.histogram(v, bins=bins, range=(lo, hi), density=True)
    w, h = width, int(height)
    ml, mt, mb = 50.0, 36.0, 44.0
    maxh = h - mt - mb
    ih = maxh
    bar_w = (w - ml - 20) / bins
    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{w/2}" y="20" text-anchor="middle" font-size="14">Distribution: {html.escape(param)} (final gen)</text>',
    ]
    mx = float(np.max(hist)) or 1.0
    for i, c in enumerate(hist):
        bh = (c / mx) * (ih * 0.9)
        x = ml + i * bar_w
        y = mt + ih - bh
        parts.append(
            f'<rect x="{x}" y="{y}" width="{bar_w*0.92}" height="{bh}" fill="#6baed6" stroke="#2171b5" stroke-width="0.2"/>'
        )
    parts.append(
        f'<text x="{w/2}" y="{h-12}" text-anchor="middle" font-size="11">{html.escape(param)}</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts)
