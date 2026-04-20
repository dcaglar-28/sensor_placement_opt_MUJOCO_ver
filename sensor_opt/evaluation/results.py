"""
Structured outputs for multi-objective, multi-fidelity evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from sensor_opt.loss.loss import EvalMetrics, LossResult


@dataclass
class EvaluationResult:
    metrics: EvalMetrics
    loss: LossResult
    objectives: Dict[str, float]
    fidelity: str
    evaluation_time_sec: float
