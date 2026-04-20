from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sensor_opt.encoding.config import SensorConfig, decode, encode


@dataclass
class ConfigEncoder:
    """Bridge between SensorConfig objects and numeric vectors."""

    mounting_slots: list[str]
    sensor_budget: dict

    def encode(self, config: SensorConfig) -> np.ndarray:
        return encode(config, self.mounting_slots)

    def decode(self, vector: np.ndarray) -> SensorConfig:
        return decode(vector, self.mounting_slots, self.sensor_budget)
