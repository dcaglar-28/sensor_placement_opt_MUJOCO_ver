"""
Design-level config combining sensor placement and hardware metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from sensor_opt.encoding.config import SensorConfig


@dataclass
class HardwareConfig:
    """Simple hardware descriptor and deployment budgets."""

    name: str = "default_hw"
    compute_limit_tops: float = 100.0
    memory_limit_gb: float = 16.0
    latency_budget_ms: float = 100.0


@dataclass
class DesignConfig:
    """Unified co-design object: sensors + hardware constraints."""

    sensors: SensorConfig
    hardware: HardwareConfig = field(default_factory=HardwareConfig)


def build_design_config(config: SensorConfig, cfg: Optional[dict] = None) -> DesignConfig:
    """Construct a DesignConfig from SensorConfig and optional experiment cfg."""
    hw_cfg: Dict[str, float] = (cfg or {}).get("hardware", {})
    hardware = HardwareConfig(
        name=hw_cfg.get("name", "default_hw"),
        compute_limit_tops=float(hw_cfg.get("compute_limit_tops", 100.0)),
        memory_limit_gb=float(hw_cfg.get("memory_limit_gb", 16.0)),
        latency_budget_ms=float(hw_cfg.get("latency_budget_ms", 100.0)),
    )
    return DesignConfig(sensors=config, hardware=hardware)
