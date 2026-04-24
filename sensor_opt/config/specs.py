"""Validation and normalization for sensor, quantity, and hardware specs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Iterable, List

from sensor_opt.config.catalog import apply_sensor_catalog

ISAAC_SIM_REQUIRED_HARDWARE_FIELDS = (
    "gpu_cores",
    "unified_memory_gb",
    "memory_bandwidth_gbps",
)

SENSOR_NUMERIC_FIELDS = (
    "cost_usd",
    "range_m",
    "horizontal_fov_deg",
    "vertical_fov_deg",
    "mass_kg",
    "compute_tops",
    "memory_gb",
    "latency_ms",
)


def prepare_experiment_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    End-to-end config prep: catalog → `sensor_budget` normalize → validate.

    Use this from `load_config` and from notebooks so `usermax` and `max_count`
    stay in sync.
    """
    out = deepcopy(apply_sensor_catalog(cfg))
    normalize_sensor_budget_inplace(out)
    validate_experiment_specs(out)
    return out


def normalize_sensor_budget_inplace(cfg: Dict[str, Any]) -> None:
    """
    In-place: copy `usermax` (max units available for that sensor type) to
    `max_count` when `max_count` is omitted so the CMA-ES encoder sees a single cap.

    If both are set, they must be equal. The optimizer is still free to use
    *fewer* than that many: inactive slots are represented as `disabled` types.
    """
    budget = cfg.get("sensor_budget")
    if not isinstance(budget, dict):
        return
    for t, spec in budget.items():
        if not isinstance(spec, dict):
            continue
        um = spec.get("usermax", None)
        mc = spec.get("max_count", None)
        if um is not None and mc is not None:
            if int(um) != int(mc):
                raise ValueError(
                    f"sensor_budget[{t!r}]: usermax and max_count must be equal "
                    f"when both are set ({um} != {mc})"
                )
        if um is not None and mc is None:
            spec["max_count"] = int(um)
        if spec.get("max_count") is None and spec.get("usermax") is None:
            raise ValueError(
                f"sensor_budget[{t!r}]: set usermax (max units you can use) and/or max_count"
            )


def quantity_values(sensor_budget: Dict[str, Dict[str, Any]], sensor_type: str) -> List[int]:
    """
    For tests: return each integer *count* of active sensors of this type that the
    budget allows (inclusive), from `min_count` (default 0) through `max_count`
    (after any `usermax` → `max_count` normalize).
    """
    spec = sensor_budget.get(sensor_type)
    if not isinstance(spec, dict):
        raise KeyError(f"Unknown sensor type in sensor_budget: {sensor_type!r}")
    lo, hi = _quantity_bounds(sensor_type, spec)
    return list(range(lo, hi + 1))


def validate_experiment_specs(cfg: Dict[str, Any]) -> None:
    """
    Validate user-entered numeric specs.

    Hardware fields such as gpu_cores / unified_memory_gb / memory_bandwidth_gbps
    are required only for bridge-backed Isaac Sim runs.
    """
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")

    _validate_sensor_budget(cfg.get("sensor_budget", {}))
    _validate_sensor_models(cfg.get("sensor_models", {}))

    mode = str((cfg.get("inner_loop") or {}).get("mode", "")).lower()
    if mode == "isaac_sim":
        _validate_isaac_hardware(cfg.get("hardware"))


def _validate_sensor_budget(sensor_budget: Any) -> None:
    if not isinstance(sensor_budget, dict) or not sensor_budget:
        raise ValueError("sensor_budget must be a non-empty mapping")
    for sensor_type, spec in sensor_budget.items():
        if not isinstance(spec, dict):
            raise ValueError(f"sensor_budget[{sensor_type!r}] must be a mapping")
        _quantity_bounds(str(sensor_type), spec)


def _quantity_bounds(sensor_type: str, spec: Dict[str, Any]) -> tuple[int, int]:
    if "max_count" not in spec:
        raise ValueError(
            f"sensor_budget[{sensor_type!r}].max_count is required (or set usermax only; "
            f"call prepare_experiment_config / normalize_sensor_budget_inplace first)"
        )
    lo = int(spec.get("min_count", 0) or 0)
    hi = int(spec["max_count"])
    if lo < 0 or hi < 0:
        raise ValueError(f"sensor_budget[{sensor_type!r}] counts must be non-negative")
    if lo > hi:
        raise ValueError(
            f"sensor_budget[{sensor_type!r}].min_count cannot be greater than max_count"
        )
    return lo, hi


def _validate_sensor_models(sensor_models: Any) -> None:
    if not isinstance(sensor_models, dict):
        raise ValueError("sensor_models must be a mapping")
    for sensor_type, model in sensor_models.items():
        if not isinstance(model, dict):
            raise ValueError(f"sensor_models[{sensor_type!r}] must be a mapping")
        _require_numeric_when_present(
            f"sensor_models[{sensor_type!r}]", model, SENSOR_NUMERIC_FIELDS
        )


def _validate_isaac_hardware(hardware: Any) -> None:
    if not isinstance(hardware, dict):
        raise ValueError("hardware specs are required when inner_loop.mode is 'isaac_sim'")
    _require_numeric("hardware", hardware, ISAAC_SIM_REQUIRED_HARDWARE_FIELDS)


def _require_numeric_when_present(
    label: str, data: Dict[str, Any], fields: Iterable[str]
) -> None:
    for field in fields:
        if field in data and data[field] is not None:
            _as_positive_number(label, field, data[field], allow_zero=True)


def _require_numeric(label: str, data: Dict[str, Any], fields: Iterable[str]) -> None:
    missing = [f for f in fields if f not in data or data[f] is None]
    if missing:
        raise ValueError(f"{label} missing required numeric fields: {', '.join(missing)}")
    for field in fields:
        _as_positive_number(label, field, data[field], allow_zero=False)


def _as_positive_number(label: str, field: str, value: Any, *, allow_zero: bool) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}.{field} must be numeric") from exc
    if allow_zero:
        if out < 0.0:
            raise ValueError(f"{label}.{field} must be >= 0")
    elif out <= 0.0:
        raise ValueError(f"{label}.{field} must be > 0")
    return out
