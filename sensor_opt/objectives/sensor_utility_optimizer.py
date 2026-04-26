"""
Utility fitness for CMA-ES sensor selection.

Implements:
  U_r = sum(IG_i) / (sum(cost_usd_i) + sum(compute_tops_i))

Where:
  IG_i = (range_i / max_range) * ((hfov_i * vfov_i) / (360 * 180))

`evaluate(...)` returns -U_r because CMA-ES minimizes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import yaml

from sensor_opt.encoding.config import SensorConfig


Penalty = float


@dataclass(frozen=True)
class SensorUtilityOptimizer:
    """
    Computes a scalar utility for a chosen sensor set.

    Notes:
    - Enforces a hard compute TOPS limit using `hardware.compute_limit_tops`.
    - Enforces per-type quantity caps using `sensor_budget.<type>.usermax` (or max_count).
    - Assumes `sensor_models` contains per-type:
        cost_usd, compute_tops, range_m, horizontal_fov_deg, vertical_fov_deg
    """

    sensor_models: Dict[str, Dict[str, float]]
    usermax: Dict[str, int]
    compute_limit_tops: float
    max_range_m: float = 100.0
    penalty_value: Penalty = 1e9

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "SensorUtilityOptimizer":
        sm_raw = cfg.get("sensor_models", {}) or {}
        if not isinstance(sm_raw, dict):
            raise TypeError("config['sensor_models'] must be a mapping")

        sensor_models: Dict[str, Dict[str, float]] = {}
        for t, model in sm_raw.items():
            if not isinstance(model, dict):
                continue
            sensor_models[str(t)] = {
                "cost_usd": float(model.get("cost_usd", 0.0) or 0.0),
                "compute_tops": float(model.get("compute_tops", 0.0) or 0.0),
                "range_m": float(model.get("range_m", 0.0) or 0.0),
                "horizontal_fov_deg": float(model.get("horizontal_fov_deg", 0.0) or 0.0),
                "vertical_fov_deg": float(model.get("vertical_fov_deg", 0.0) or 0.0),
            }

        # Per-type caps (preferred: sensor_budget.<type>.usermax)
        budget = cfg.get("sensor_budget", {}) or {}
        if budget is not None and not isinstance(budget, dict):
            raise TypeError("config['sensor_budget'] must be a mapping when present")
        usermax: Dict[str, int] = {}
        if isinstance(budget, dict):
            for t, spec in budget.items():
                if not isinstance(spec, dict):
                    continue
                if spec.get("usermax") is not None:
                    usermax[str(t)] = int(spec["usermax"])
                elif spec.get("max_count") is not None:
                    usermax[str(t)] = int(spec["max_count"])

        hw = cfg.get("hardware", {}) or {}
        if not isinstance(hw, dict):
            raise TypeError("config['hardware'] must be a mapping")
        compute_limit = float(hw.get("compute_limit_tops", 0.0) or 0.0)
        if compute_limit <= 0.0:
            raise ValueError("hardware.compute_limit_tops must be > 0")

        # Max range normalization: per your requirement, fixed at 100.0 (lidar range_m in YAML).
        max_range = 100.0

        loss_cfg = cfg.get("loss", {}) or {}
        if loss_cfg is not None and not isinstance(loss_cfg, dict):
            raise TypeError("config['loss'] must be a mapping when present")
        penalty_value = float(loss_cfg.get("penalty_value", 1e9) or 1e9) if isinstance(loss_cfg, dict) else 1e9

        return cls(
            sensor_models=sensor_models,
            usermax=usermax,
            compute_limit_tops=compute_limit,
            max_range_m=float(max_range),
            penalty_value=penalty_value,
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, "os.PathLike[str]"]) -> "SensorUtilityOptimizer":
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            raise TypeError("YAML root must be a mapping")
        return cls.from_config(cfg)

    def evaluate(self, config: Union[SensorConfig, Sequence[str]]) -> float:
        """
        Returns negative utility (-U_r) for CMA-ES minimization.

        Args:
          config:
            - SensorConfig (preferred), or
            - sequence of sensor type strings (e.g. ['camera','radar',...]) for quick use.
        """
        util = self.utility_ratio(config)
        return -float(util) if util < self.penalty_value else float(self.penalty_value)

    def utility_ratio(self, config: Union[SensorConfig, Sequence[str]]) -> float:
        types = self._extract_types(config)
        ok, violated = self._check_usermax(types)
        if not ok:
            return self.penalty_value

        cost_sum = 0.0
        compute_sum = 0.0
        ig_sum = 0.0

        for t in types:
            if t == "disabled":
                continue
            m = self.sensor_models.get(t, {})
            cost_sum += float(m.get("cost_usd", 0.0) or 0.0)
            compute_sum += float(m.get("compute_tops", 0.0) or 0.0)
            ig_sum += self._information_gain(m)

        if compute_sum > self.compute_limit_tops + 1e-9:
            return self.penalty_value

        denom = cost_sum + compute_sum
        if denom <= 1e-12:
            return self.penalty_value

        return float(ig_sum / denom)

    def _information_gain(self, model: Mapping[str, float]) -> float:
        r = float(model.get("range_m", 0.0) or 0.0)
        hfov = float(model.get("horizontal_fov_deg", 0.0) or 0.0)
        vfov = float(model.get("vertical_fov_deg", 0.0) or 0.0)

        r_n = 0.0 if self.max_range_m <= 0.0 else max(0.0, min(1.0, r / self.max_range_m))
        area_n = max(0.0, min(1.0, (hfov * vfov) / (360.0 * 180.0)))
        return float(r_n * area_n)

    def _extract_types(self, config: Union[SensorConfig, Sequence[str]]) -> Sequence[str]:
        if isinstance(config, SensorConfig):
            return [s.sensor_type for s in config.sensors]
        return [str(t) for t in config]

    def _check_usermax(self, types: Sequence[str]) -> Tuple[bool, Optional[Dict[str, int]]]:
        if not self.usermax:
            return True, None
        counts: Dict[str, int] = {}
        violated: Dict[str, int] = {}
        for t in types:
            if t == "disabled":
                continue
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            lim = self.usermax.get(t)
            if lim is not None and c > int(lim):
                violated[t] = c
        return (len(violated) == 0), (violated if violated else None)

