from copy import deepcopy

from sensor_opt.config.colab_builtins import (
    ISAAC_SAFETY,
    apply_safety_guards_experiment_config,
    get_default_colab_config,
    parse_float_user,
    parse_int_user,
    prompt_isaac_hardware_only,
    prompt_sensor_budget_usermax,
)
from sensor_opt.config.specs import prepare_experiment_config


def test_get_default_colab_config_loads_and_validates():
    c = get_default_colab_config()
    assert c["inner_loop"]["mode"] == "isaac_sim"
    assert c["loss"]["mode"] == "obstacle_latency"
    p = prepare_experiment_config(deepcopy(c))
    assert p["sensor_budget"]["lidar"]["max_count"] >= 0


def test_parse_int_user_empty_uses_def():
    spec = {"def": 7, "min": 0, "max": 10}
    v, fb, reas = parse_int_user("", spec=spec)
    assert v == 7 and fb and reas == "empty"
    v2, fb2, _ = parse_int_user("3", spec=spec)
    assert v2 == 3 and not fb2


def test_parse_int_user_oob_uses_def():
    spec = ISAAC_SAFETY["hardware"]["gpu_cores"]
    v, fb, reas = parse_int_user("9999999999", spec=spec)
    assert v == int(spec["def"]) and fb and reas and "range" in reas


def test_parse_float_user_nonfinite_uses_def():
    spec = ISAAC_SAFETY["isaac_sim"]["sensor_noise_std"]
    v, fb, reas = parse_float_user("nan", spec=spec)
    assert v == float(spec["def"]) and fb


def test_prompt_sensor_budget_no_input_all_defaults(monkeypatch):
    n = 0

    def fake(_):
        nonlocal n
        n += 1
        return ""  # always fall back to def

    monkeypatch.setattr("builtins.input", fake)
    raw: dict = {
        "sensor_budget": {
            "lidar": {"max_count": 99},
            "camera": {"usermax": 1},
        }
    }
    prompt_sensor_budget_usermax(raw)
    udef = ISAAC_SAFETY["sensor_budget"]["default_usermax"]
    d = int(udef["def"])
    assert raw["sensor_budget"]["lidar"]["usermax"] == d
    assert raw["sensor_budget"]["camera"]["usermax"] == d
    assert n == 2


def test_apply_safety_guards_applies_sensor_env_overrides(monkeypatch):
    monkeypatch.setenv("SENSOR_LIDAR_MAX", "3")
    monkeypatch.setenv("SENSOR_CAMERA_MIN", "1")
    raw = {
        "inner_loop": {"mode": "isaac_sim", "n_episodes": 1, "max_steps_per_episode": 1, "isaac_sim": {"sensor_noise_std": 0.0}},
        "sensor_budget": {
            "lidar": {"usermax": 0},
            "camera": {"usermax": 2},
            "radar": {"usermax": 0},
        },
        "loss": {"mode": "obstacle_latency", "max_cost_usd": 1.0},
        "cma": {"max_generations": 1, "population_size": 2, "sigma0": 0.1},
        "experiment": {"name": "e", "seed": 0},
        "hardware": {"gpu_cores": 1, "unified_memory_gb": 1.0, "memory_bandwidth_gbps": 1.0},
    }
    apply_safety_guards_experiment_config(raw)
    assert raw["sensor_budget"]["lidar"]["usermax"] == 3
    assert raw["sensor_budget"]["camera"]["min_count"] == 1


def test_apply_safety_guards_recovers_bad_hardware():
    raw = {
        "inner_loop": {"mode": "isaac_sim", "n_episodes": -1, "max_steps_per_episode": 0, "isaac_sim": {"sensor_noise_std": 1e9}},
        "sensor_budget": {"lidar": {"usermax": 9999}},
        "loss": {"mode": "obstacle_latency", "alpha": -1, "beta": -1, "max_cost_usd": -1, "t_det_max_s": 0},
        "cma": {"max_generations": 0, "population_size": 0, "sigma0": 0.0},
        "experiment": {"name": "   ", "seed": "bad"},
        "hardware": {"gpu_cores": 0, "unified_memory_gb": -1, "memory_bandwidth_gbps": 0.0},
    }
    apply_safety_guards_experiment_config(raw)
    assert int(raw["hardware"]["gpu_cores"]) >= 1
    assert float(raw["hardware"]["unified_memory_gb"]) > 0.0
    p = prepare_experiment_config(deepcopy(raw))
    assert p["inner_loop"]["n_episodes"] >= 1


def test_prompt_isaac_hardware_int_floats_and_cost_budget(monkeypatch):
    # gpu (int), unified (GB), loss max_cost_usd (float); YAML defaults for bandwidth/name
    lines = iter(["1200", "32", "750.0"])

    def fake(_p):
        return next(lines)

    monkeypatch.setattr("builtins.input", fake)
    raw: dict = {"inner_loop": {"mode": "isaac_sim"}}
    prompt_isaac_hardware_only(raw)
    assert raw["hardware"]["gpu_cores"] == 1200
    assert raw["hardware"]["unified_memory_gb"] == 32.0
    assert raw["hardware"]["memory_bandwidth_gbps"] == 320.0
    assert raw["loss"]["max_cost_usd"] == 750.0
