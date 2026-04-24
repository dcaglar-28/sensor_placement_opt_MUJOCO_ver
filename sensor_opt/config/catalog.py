"""
sensor_opt/config/catalog.py

Optional config pre-processing to support a "sensor_catalog" + "sensor_choices"
YAML style while keeping the rest of the codebase unchanged.

The optimizer/search stack expects:
  cfg["sensor_models"] = { "<sensor_type>": { ...model fields... }, ... }

This module allows users to instead provide:
  cfg["sensor_catalog"] = { "<model_id>": { sensor_type: "lidar", ... }, ... }
  cfg["sensor_choices"] = { "lidar": "<model_id>", "camera": "<model_id>", ... }

and we synthesize cfg["sensor_models"] accordingly.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


def apply_sensor_catalog(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of cfg where `sensor_models` is synthesized from catalog inputs.

    Backward compatible:
    - If cfg already has `sensor_models`, this is a no-op.
    - If cfg has no `sensor_catalog`, this is a no-op.
    """

    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")

    if "sensor_models" in cfg and cfg["sensor_models"] is not None:
        return cfg

    if "sensor_catalog" not in cfg:
        return cfg

    catalog = cfg.get("sensor_catalog")
    choices = cfg.get("sensor_choices")

    if not isinstance(catalog, dict) or not catalog:
        raise ValueError("`sensor_catalog` must be a non-empty mapping")
    if not isinstance(choices, dict) or not choices:
        raise ValueError("`sensor_choices` must be a non-empty mapping (e.g., {lidar: vlp16, camera: d435i})")

    sensor_models: Dict[str, Dict[str, Any]] = {}
    for sensor_type, model_id in choices.items():
        if not isinstance(sensor_type, str) or not sensor_type:
            raise ValueError("`sensor_choices` keys must be non-empty strings (sensor types)")
        if not isinstance(model_id, str) or not model_id:
            raise ValueError("`sensor_choices` values must be non-empty strings (catalog model ids)")
        if model_id not in catalog:
            raise KeyError(f"`sensor_choices[{sensor_type!r}]` references unknown model_id {model_id!r}")

        model = deepcopy(catalog[model_id])
        if not isinstance(model, dict):
            raise ValueError(f"`sensor_catalog[{model_id!r}]` must be a mapping")

        declared_type = model.get("sensor_type", model.get("type"))
        if declared_type is not None and str(declared_type) != str(sensor_type):
            raise ValueError(
                f"Catalog model {model_id!r} declares sensor_type={declared_type!r} "
                f"but it is selected for {sensor_type!r}"
            )

        # Remove catalog-only metadata; the rest of the code expects pure model fields.
        model.pop("sensor_type", None)
        model.pop("type", None)

        sensor_models[str(sensor_type)] = model

    out = deepcopy(cfg)
    out["sensor_models"] = sensor_models
    return out

