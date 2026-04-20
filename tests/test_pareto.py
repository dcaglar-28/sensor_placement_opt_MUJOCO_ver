"""tests/test_pareto.py"""

from sensor_opt.cma.pareto import dominates, pareto_front


def test_dominates_minimization():
    assert dominates({"collision": 0.1, "cost": 0.3}, {"collision": 0.2, "cost": 0.3})
    assert not dominates({"collision": 0.2, "cost": 0.3}, {"collision": 0.1, "cost": 0.3})


def test_pareto_front_non_dominated_selection():
    configs = ["a", "b", "c"]
    results = [
        {"collision": 0.2, "blind_spot": 0.2, "cost": 0.2},
        {"collision": 0.1, "blind_spot": 0.3, "cost": 0.1},
        {"collision": 0.3, "blind_spot": 0.3, "cost": 0.3},
    ]
    front = pareto_front(configs, results)
    idxs = sorted([p.index for p in front])
    assert idxs == [0, 1]
