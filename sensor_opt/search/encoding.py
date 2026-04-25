from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from sensor_opt.encoding.config import SensorConfig, decode, encode


@dataclass
class ConfigEncoder:
    """Bridge between SensorConfig objects and numeric vectors."""

    mounting_slots: list[str]
    sensor_budget: dict
    fixed_mount_order: bool = False
    fixed_sensor_geometry: bool = False
    default_sensor_pose: Optional[dict[str, Any]] = None

    def encode(self, config: SensorConfig) -> np.ndarray:
        fmo = self.fixed_mount_order or self.fixed_sensor_geometry
        return encode(
            config,
            self.mounting_slots,
            fixed_mount_order=fmo,
            fixed_sensor_geometry=self.fixed_sensor_geometry,
        )

    def decode(self, vector: np.ndarray) -> SensorConfig:
        fmo = self.fixed_mount_order or self.fixed_sensor_geometry
        return decode(
            vector,
            self.mounting_slots,
            self.sensor_budget,
            fixed_mount_order=fmo,
            fixed_sensor_geometry=self.fixed_sensor_geometry,
            default_sensor_pose=self.default_sensor_pose,
        )


def make_config_encoder(cfg) -> "ConfigEncoder":
    """Build encoder from a full experiment `cfg` (YAML / dict)."""
    return ConfigEncoder(
        mounting_slots=cfg["mounting_slots"],
        sensor_budget=cfg["sensor_budget"],
        fixed_mount_order=bool(cfg.get("fixed_mount_order", False)),
        fixed_sensor_geometry=bool(cfg.get("fixed_sensor_geometry", False)),
        default_sensor_pose=cfg.get("default_sensor_pose", {}) or {},
    )
