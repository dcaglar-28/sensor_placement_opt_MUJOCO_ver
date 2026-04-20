"""
Pareto-front utilities for multi-objective co-design.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


@dataclass
class ParetoPoint:
    config: object
    objectives: Dict[str, float]
    index: int


def dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    """
    True if a Pareto-dominates b (assuming minimization for all objectives).
    """
    keys = sorted(set(a.keys()) & set(b.keys()))
    if not keys:
        return False
    no_worse = all(a[k] <= b[k] for k in keys)
    strictly_better = any(a[k] < b[k] for k in keys)
    return no_worse and strictly_better


def pareto_front(configs: Sequence[object], results: Sequence[Dict[str, float]]) -> List[ParetoPoint]:
    """
    Return non-dominated (config, objective) points.
    """
    points = [ParetoPoint(config=c, objectives=r, index=i) for i, (c, r) in enumerate(zip(configs, results))]
    front: List[ParetoPoint] = []
    for candidate in points:
        is_dominated = False
        for other in points:
            if other.index == candidate.index:
                continue
            if dominates(other.objectives, candidate.objectives):
                is_dominated = True
                break
        if not is_dominated:
            front.append(candidate)
    return front
