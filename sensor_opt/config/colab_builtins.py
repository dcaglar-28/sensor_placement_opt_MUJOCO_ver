"""
Colab: bundled obstacle+Isaac template, validated prompts, and a safety net so
invalid/empty user input does not break Isaac or `prepare_experiment_config`.
"""

from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import yaml

RepoDict = Dict[str, Any]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OBSTACLE_ISAACLAB_YAML = _REPO_ROOT / "configs" / "obstacle_isaaclab.yaml"

# Valid ranges and fallbacks (Isaac + sim-friendly). Used for prompts and for guards.
ISAAC_SAFETY: Dict[str, Any] = {
    "hardware": {
        "gpu_cores": {"def": 2560, "min": 1, "max": 1_000_000},
        "unified_memory_gb": {"def": 16.0, "min": 0.5, "max": 8192.0},
        "memory_bandwidth_gbps": {"def": 320.0, "min": 0.1, "max": 20_000.0},
        # Sensor placement budget (USD), mirrors `loss.max_cost_usd` bounds for Colab defaults.
        "cost_budget_usd": {"def": 10_000.0, "min": 0.0, "max": 1.0e15},
    },
    "inner_loop": {
        "n_episodes": {"def": 20, "min": 1, "max": 1_000_000},
        "max_steps_per_episode": {"def": 500, "min": 1, "max": 1_000_000},
    },
    "isaac_sim": {
        "sensor_noise_std": {"def": 0.0, "min": 0.0, "max": 100.0},
    },
    "cma": {
        "max_generations": {"def": 100, "min": 1, "max": 1_000_000},
        "population_size": {"def": 20, "min": 2, "max": 1_000_000},
        "sigma0": {"def": 0.3, "min": 1e-6, "max": 1e6},
    },
    "loss": {
        "alpha": {"def": 1.0, "min": 0.0, "max": 1.0e12},
        "beta": {"def": 100.0, "min": 0.0, "max": 1.0e12},
        "max_cost_usd": {"def": 100_000.0, "min": 0.0, "max": 1.0e15},
        "t_det_max_s": {"def": 10.0, "min": 0.1, "max": 1.0e6},
    },
    "sensor_budget": {
        "default_usermax": {"def": 2, "min": 0, "max": 32},
    },
    "experiment": {
        "seed": {"def": 42, "min": 0, "max": 2**31 - 1},
        "name": {"def": "colab_isaaclab_run", "max_len": 200},
    },
}


def _alert(msg: str) -> None:
    print(f"[sensor_placement_opt] {msg}")


def get_default_colab_config() -> RepoDict:
    if not _OBSTACLE_ISAACLAB_YAML.is_file():
        raise FileNotFoundError(
            f"Colab default config not found: {_OBSTACLE_ISAACLAB_YAML} "
            "(use a full `sensor_placement_opt` clone with configs/)."
        )
    with open(_OBSTACLE_ISAACLAB_YAML) as f:
        return deepcopy(yaml.safe_load(f))


def _in(d: Any, *keys: str) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def parse_int_user(
    raw: Optional[str], spec: Dict[str, Union[int, float]]
) -> Tuple[int, bool, Optional[str]]:
    """(value, used_fallback, reason). Any failure → spec['def']."""
    d = int(spec["def"])
    lo, hi = int(spec["min"]), int(spec["max"])
    s = (raw or "").strip()
    if not s:
        return d, True, "empty"
    try:
        v = int(s, 0)
    except (TypeError, ValueError):
        return d, True, "not a valid integer"
    if v < lo or v > hi:
        return d, True, f"outside valid range [{lo}, {hi}]"
    return v, False, None


def parse_float_user(
    raw: Optional[str], spec: Dict[str, Union[int, float]]
) -> Tuple[float, bool, Optional[str]]:
    d = float(spec["def"])
    lo, hi = float(spec["min"]), float(spec["max"])
    s = (raw or "").strip()
    if not s:
        return d, True, "empty"
    try:
        v = float(s)
    except (TypeError, ValueError):
        return d, True, "not a valid number"
    if not math.isfinite(v):
        return d, True, "non-finite (inf/nan)"
    if v < lo or v > hi:
        return d, True, f"outside valid range [{lo}, {hi}]"
    return v, False, None


def _read_int(field_label: str, prompt_name: str, spec: Dict[str, Union[int, float]]) -> int:
    try:
        s = input(
            f"{prompt_name}  (valid {int(spec['min'])}..{int(spec['max'])}; safe default {int(spec['def'])}): "
        ).strip()
    except EOFError:
        s = ""
    v, used_fb, reason = parse_int_user(s, spec)
    if used_fb:
        _alert(f"{field_label}: {reason} — using safe default {v}.")
    return v


def _read_float(field_label: str, prompt_name: str, spec: Dict[str, Union[int, float]]) -> float:
    try:
        s = input(
            f"{prompt_name}  (valid {float(spec['min'])}..{float(spec['max'])}; safe default {float(spec['def'])}): "
        ).strip()
    except EOFError:
        s = ""
    v, used_fb, reason = parse_float_user(s, spec)
    if used_fb:
        _alert(f"{field_label}: {reason} — using safe default {v}.")
    return v


def _read_str_safety(
    field_label: str, prompt_name: str, *, default: str, max_len: int
) -> str:
    """Empty input uses `default` (no reuse of file/template strings). Keystrokes only count as the result."""
    try:
        s = input(f"{prompt_name}  (empty → safe default {default!r}): ").strip()
    except EOFError:
        s = ""
    if not s:
        _alert(f"{field_label}: empty — using safe default {default!r}.")
        return default
    if len(s) > max_len:
        _alert(f"{field_label}: input truncated to {max_len} characters.")
        return s[:max_len]
    return s


def _apply_sensor_budget_from_env(raw: RepoDict) -> None:
    """If SENSOR_*_MAX / SENSOR_*_MIN are set, copy into `sensor_budget` (Colab)."""
    sb = raw.get("sensor_budget")
    if not isinstance(sb, dict):
        return
    udef = ISAAC_SAFETY["sensor_budget"]["default_usermax"]
    mnspec: Dict[str, Any] = {"def": 0, "min": 0, "max": int(udef["max"])}
    for emax, emin, stype in (
        ("SENSOR_LIDAR_MAX", "SENSOR_LIDAR_MIN", "lidar"),
        ("SENSOR_CAMERA_MAX", "SENSOR_CAMERA_MIN", "camera"),
        ("SENSOR_RADAR_MAX", "SENSOR_RADAR_MIN", "radar"),
    ):
        if stype not in sb or not isinstance(sb[stype], dict):
            continue
        sp0 = sb[stype]
        if emax in os.environ:
            s = os.environ[emax].strip()
            n = parse_int_user(s, udef)[0]
            sp0["usermax"] = int(n)
            sp0.pop("max_count", None)
        if emin in os.environ:
            s = os.environ[emin].strip()
            m = parse_int_user(s, mnspec)[0]
            sp0["min_count"] = int(m)


def apply_safety_guards_experiment_config(raw: RepoDict) -> None:
    """
    In-place: enforce ranges and required keys after prompts or external edits
    so `prepare_experiment_config` and the bridge do not see broken values.
    """
    _apply_sensor_budget_from_env(raw)
    is_isaac = str(_in(raw, "inner_loop", "mode") or "").lower() == "isaac_sim"
    if is_isaac:
        h = raw.setdefault("hardware", {})
        if not isinstance(h, dict):
            raw["hardware"] = {
                "name": "colab_safety",
            }
            h = raw["hardware"]
        gspec, uspec, mspec = (
            ISAAC_SAFETY["hardware"]["gpu_cores"],
            ISAAC_SAFETY["hardware"]["unified_memory_gb"],
            ISAAC_SAFETY["hardware"]["memory_bandwidth_gbps"],
        )
        h["gpu_cores"] = int(
            parse_int_user(str(int(h.get("gpu_cores", 0) or 0)), gspec)[0]
        )
        h["unified_memory_gb"] = float(
            parse_float_user(str(h.get("unified_memory_gb", 0) or 0.0), uspec)[0]
        )
        h["memory_bandwidth_gbps"] = float(
            parse_float_user(str(h.get("memory_bandwidth_gbps", 0) or 0.0), mspec)[0]
        )
        h.setdefault("compute_limit_tops", 1.0e9)
        h.setdefault("memory_limit_gb", 1.0e9)
        h.setdefault("latency_budget_ms", 1.0e9)
        h.setdefault("name", "colab_safety")

    sb = raw.get("sensor_budget")
    udef = ISAAC_SAFETY["sensor_budget"]["default_usermax"]
    mnspec2: Dict[str, Any] = {"def": 0, "min": 0, "max": int(udef["max"])}
    if isinstance(sb, dict):
        for t, sp0 in list(sb.items()):
            if not isinstance(sp0, dict):
                continue
            rawv = sp0.get("usermax", sp0.get("max_count", udef["def"]))
            try:
                raws = str(int(rawv) if rawv is not None else int(udef["def"]))
            except (TypeError, ValueError):
                raws = str(int(udef["def"]))
            n, fb, reas = parse_int_user(raws, udef)
            if fb:
                _alert(f"apply_safety_guards: sensor_budget[{t!r}].usermax = {n} ({reas}).")
            sp0["usermax"] = int(n)
            sp0.pop("max_count", None)
            mraw = str(int(sp0.get("min_count", 0) or 0))
            mn, fb2, reas2 = parse_int_user(mraw, mnspec2)
            if int(mn) > int(n):
                mn = n
                fb2 = True
                reas2 = "min_count > usermax"
            if fb2:
                _alert(f"apply_safety_guards: sensor_budget[{t!r}].min_count = {mn} ({reas2}).")
            sp0["min_count"] = int(mn)

    il = raw.get("inner_loop")
    in_ep, stp = ISAAC_SAFETY["inner_loop"]["n_episodes"], ISAAC_SAFETY["inner_loop"]["max_steps_per_episode"]
    if isinstance(il, dict) and str(il.get("mode", "")).lower() == "isaac_sim":
        il["n_episodes"] = int(parse_int_user(str(il.get("n_episodes", 0) or 0), in_ep)[0])
        il["max_steps_per_episode"] = int(
            parse_int_user(str(il.get("max_steps_per_episode", 0) or 0), stp)[0]
        )
        isim = il.setdefault("isaac_sim", {})
        if not isinstance(isim, dict):
            isim = {}
            il["isaac_sim"] = isim
        sp0 = ISAAC_SAFETY["isaac_sim"]["sensor_noise_std"]
        isim["sensor_noise_std"] = float(
            parse_float_user(str(isim.get("sensor_noise_std", 0) or 0.0), sp0)[0]
        )

    cma = raw.get("cma")
    if isinstance(cma, dict):
        cs = ISAAC_SAFETY["cma"]
        cma["max_generations"] = int(
            parse_int_user(str(cma.get("max_generations", 0) or 0), cs["max_generations"])[0]
        )
        cma["population_size"] = int(
            parse_int_user(str(cma.get("population_size", 0) or 0), cs["population_size"])[0]
        )
        cma["sigma0"] = float(
            parse_float_user(str(cma.get("sigma0", 0) or 0.0), cs["sigma0"])[0]
        )

    lo = raw.get("loss")
    if isinstance(lo, dict):
        ls = ISAAC_SAFETY["loss"]
        lo["max_cost_usd"] = float(
            parse_float_user(str(lo.get("max_cost_usd", 0) or 0.0), ls["max_cost_usd"])[0]
        )
        if "t_det_max_s" in lo:
            lo["t_det_max_s"] = float(
                parse_float_user(str(lo.get("t_det_max_s", 0) or 0.0), ls["t_det_max_s"])[0]
            )
        if str(lo.get("mode", "")).lower() == "obstacle_latency":
            lo["alpha"] = float(
                parse_float_user(str(lo.get("alpha", 0) or 0.0), ls["alpha"])[0]
            )
            lo["beta"] = float(
                parse_float_user(str(lo.get("beta", 0) or 0.0), ls["beta"])[0]
            )

    ex = raw.get("experiment")
    if isinstance(ex, dict):
        es = ISAAC_SAFETY["experiment"]
        ex["seed"] = int(
            parse_int_user(str(ex.get("seed", 0) or 0), es["seed"])[0]  # type: ignore[index]
        )
        nmax = int(es["name"].get("max_len", 200) or 200)  # type: ignore[union-attr]
        nm = ex.get("name", "")
        if not isinstance(nm, str) or not (nm and nm.strip()) or len(nm) > nmax:
            dnm = str(es["name"]["def"])
            _alert(f"apply_safety_guards: experiment.name set to {dnm!r}.")
            ex["name"] = dnm[:nmax] if len(dnm) > nmax else dnm
        elif len(nm) > nmax:
            ex["name"] = nm[:nmax]
            _alert("apply_safety_guards: experiment.name truncated.")


def prompt_sensor_budget_usermax(raw: RepoDict) -> None:
    sb = raw.get("sensor_budget")
    if not isinstance(sb, dict):
        return
    udef = ISAAC_SAFETY["sensor_budget"]["default_usermax"]
    for t, spec0 in sb.items():
        if not isinstance(spec0, dict):
            continue
        try:
            sugg = int(spec0.get("usermax", spec0.get("max_count", udef["def"])))
        except (TypeError, ValueError):
            sugg = int(udef["def"])
        n = _read_int(
            f"sensor_budget.{t}.usermax",
            f"Max {t} units in inventory  (suggested: {sugg}, safe if invalid: {udef['def']})",
            udef,
        )
        spec0["usermax"] = int(n)
        spec0.pop("max_count", None)


def _default_int_for_prompt(
    env_key: str, h: Dict[str, Any], hkey: str, gspec: Dict[str, Any]
) -> Dict[str, Any]:
    s = os.environ.get(env_key, "").strip()
    if s:
        d = int(parse_int_user(s, gspec)[0])
    else:
        v = h.get(hkey)
        if v is not None:
            d = int(parse_int_user(str(int(v)), gspec)[0])
        else:
            d = int(gspec["def"])
    return {**gspec, "def": d}


def _default_float_for_prompt(
    env_key: str, h: Dict[str, Any], hkey: str, fspec: Dict[str, Any]
) -> Dict[str, Any]:
    s = os.environ.get(env_key, "").strip()
    if s:
        d = float(parse_float_user(s, fspec)[0])
    else:
        v = h.get(hkey)
        if v is not None:
            d = float(parse_float_user(str(v), fspec)[0])
        else:
            d = float(fspec["def"])
    return {**fspec, "def": d}


def _default_loss_cost_for_prompt(raw: RepoDict) -> Dict[str, Any]:
    lspec0 = ISAAC_SAFETY["loss"]["max_cost_usd"]
    cbd = ISAAC_SAFETY["hardware"]["cost_budget_usd"]
    lo = raw.get("loss")
    s = os.environ.get("HW_COST_BUDGET_USD", "").strip()
    if s:
        d = float(parse_float_user(s, lspec0)[0])
    elif isinstance(lo, dict) and lo.get("max_cost_usd") is not None:
        d = float(parse_float_user(str(lo.get("max_cost_usd", 0) or 0.0), lspec0)[0])
    else:
        d = float(cbd["def"])
    return {**lspec0, "def": d}


def prompt_isaac_hardware_only(raw: RepoDict) -> None:
    """Prompt only gpu_cores, unified_memory_gb, and loss `max_cost_usd` (Colab / Isaac)."""
    h = raw.setdefault("hardware", {})
    if not isinstance(h, dict):
        raw["hardware"] = {}
        h = raw["hardware"]
    gspec = _default_int_for_prompt("HW_GPU_CORES", h, "gpu_cores", ISAAC_SAFETY["hardware"]["gpu_cores"])
    h["gpu_cores"] = _read_int("hardware.gpu_cores", "GPU cores (Isaac / machine)", gspec)
    uspec = _default_float_for_prompt(
        "HW_UNIFIED_MEMORY_GB", h, "unified_memory_gb", ISAAC_SAFETY["hardware"]["unified_memory_gb"]
    )
    h["unified_memory_gb"] = _read_float(
        "hardware.unified_memory_gb", "Unified memory (GB) (Isaac / machine)", uspec
    )
    h.setdefault("memory_bandwidth_gbps", float(ISAAC_SAFETY["hardware"]["memory_bandwidth_gbps"]["def"]))
    h.setdefault("name", str(ISAAC_SAFETY["experiment"]["name"]["def"]))

    lo = raw.setdefault("loss", {})
    if not isinstance(lo, dict):
        lo = {}
        raw["loss"] = lo
    cspec = _default_loss_cost_for_prompt(raw)
    lo["max_cost_usd"] = _read_float("loss.max_cost_usd", "Cost budget (USD) (sensor / run cap)", cspec)


def prompt_colab_experiment_interactive(
    raw: RepoDict, *, include_hardware: bool = True, include_cma: bool = True, include_loss: bool = True
) -> None:
    mode_is_isaac = str(_in(raw, "inner_loop", "mode") or "").lower() == "isaac_sim"
    ex0 = raw.setdefault("experiment", {})
    if not isinstance(ex0, dict):
        ex0 = {}
        raw["experiment"] = ex0
    esn = ISAAC_SAFETY["experiment"]
    nmax = int(esn["name"].get("max_len", 200) or 200)  # type: ignore[union-attr]
    dname0 = str(esn["name"]["def"])
    ex0["name"] = _read_str_safety("experiment.name", "Experiment name", default=dname0, max_len=nmax)
    ex0["seed"] = _read_int("experiment.seed", f"Random seed  (suggested: {ex0.get('seed', 42)!r})", esn["seed"])  # type: ignore[index]

    if include_hardware and mode_is_isaac:
        prompt_isaac_hardware_only(raw)

    il = raw.setdefault("inner_loop", {})
    if not isinstance(il, dict):
        il = {}
        raw["inner_loop"] = il
    in_ep = ISAAC_SAFETY["inner_loop"]
    il["n_episodes"] = _read_int(
        "inner_loop.n_episodes",
        f"Inner loop n_episodes  (suggested: {il.get('n_episodes', in_ep['n_episodes']['def'])!r})",
        in_ep["n_episodes"],
    )
    il["max_steps_per_episode"] = _read_int(
        "inner_loop.max_steps_per_episode",
        f"max_steps_per_episode  (suggested: {il.get('max_steps_per_episode', in_ep['max_steps_per_episode']['def'])!r})",
        in_ep["max_steps_per_episode"],
    )
    isim = il.setdefault("isaac_sim", {})
    if not isinstance(isim, dict):
        isim = {}
        il["isaac_sim"] = isim
    sspec = ISAAC_SAFETY["isaac_sim"]["sensor_noise_std"]
    isim["sensor_noise_std"] = _read_float(
        "inner_loop.isaac_sim.sensor_noise_std",
        f"Isaac sim sensor noise std (m) — 0 = off  (suggested: {isim.get('sensor_noise_std', 0) or 0!r})",
        sspec,
    )

    cma = raw.get("cma")
    if include_cma and isinstance(cma, dict):
        cs = ISAAC_SAFETY["cma"]
        cma["max_generations"] = _read_int("cma.max_generations", "CMA max_generations", cs["max_generations"])
        cma["population_size"] = _read_int("cma.population_size", "CMA population_size", cs["population_size"])
        cma["sigma0"] = _read_float("cma.sigma0", "CMA sigma0", cs["sigma0"])

    loss0 = raw.get("loss")
    if include_loss and isinstance(loss0, dict):
        ls = ISAAC_SAFETY["loss"]
        if str(loss0.get("mode", "")).lower() == "obstacle_latency":
            loss0["alpha"] = _read_float("loss.alpha", "Loss alpha (p95 term)", ls["alpha"])
            loss0["beta"] = _read_float("loss.beta", "Loss beta (collision term)", ls["beta"])
        if not (include_hardware and mode_is_isaac):
            loss0["max_cost_usd"] = _read_float(
                "loss.max_cost_usd", "Loss max_cost_usd (budget cap)", ls["max_cost_usd"]
            )
        if "t_det_max_s" in loss0:
            loss0["t_det_max_s"] = _read_float("loss.t_det_max_s", "Loss t_det_max_s (p95 normalizer)", ls["t_det_max_s"])


__all__ = [
    "ISAAC_SAFETY",
    "apply_safety_guards_experiment_config",
    "get_default_colab_config",
    "parse_float_user",
    "parse_int_user",
    "prompt_colab_experiment_interactive",
    "prompt_isaac_hardware_only",
    "prompt_sensor_budget_usermax",
]
