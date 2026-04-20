from __future__ import annotations

from sensor_opt.search.bayesian_search import BayesianSearch
from sensor_opt.search.cma_search import CMASearch
from sensor_opt.search.hybrid_search import HybridSearch
from sensor_opt.search.nsga2_search import NSGA2Search


def create_search(search_type, config, evaluator):
    if search_type == "cma":
        return CMASearch(config, evaluator)
    if search_type == "nsga2":
        return NSGA2Search(config, evaluator)
    if search_type == "bayesian":
        return BayesianSearch(config, evaluator)
    if search_type == "hybrid":
        return HybridSearch(config, evaluator)

    raise ValueError(f"Unknown search type: {search_type}")
