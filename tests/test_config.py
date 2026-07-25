"""Unit tests for Config loading and validation."""
import json
import os
import tempfile
import pytest

from src.config import Config, QUEUE_DEFAULTS, PIPELINE_DEFAULTS


VALID_CONFIG = {
    "coordinates": [38.98211, 38.975, -76.93006, -76.93704],
    "num_chargers": 2,
    "possible_charger_positions": [14, 20, 21],
    "od_demand": {"7,26": [60, 120]},
    "max_iter": 1000,
    "use_derivatives": False,
    "single_swap": True,
    "use_cvxpy": True,
    "plot_info": False,
    "calculate_on_all_possible_positions": True,
    "route_analysis": {"analyze_top_k_routes": True, "k_values": [1, 2, 4, 8]},
    "queue_simulation": {"K": 8, "THRESH": 500, "NUM_ITERS": 3, "N": 5},
    "pipeline": {"random_seed": 42, "skip_bpr_fitting": True}
}


def test_config_from_dict():
    cfg = Config.from_dict(VALID_CONFIG)
    assert cfg.coordinates == [38.98211, 38.975, -76.93006, -76.93704]
    assert cfg.num_chargers == 2
    assert cfg.possible_charger_positions == [14, 20, 21]
    assert cfg.queue_simulation["K"] == 8
    assert cfg.pipeline["random_seed"] == 42


def test_config_defaults_filled():
    raw = {**VALID_CONFIG}
    del raw["queue_simulation"]
    del raw["pipeline"]
    cfg = Config.from_dict(raw)
    assert cfg.queue_simulation["K"] == QUEUE_DEFAULTS["K"]
    assert cfg.pipeline["random_seed"] == PIPELINE_DEFAULTS["random_seed"]


def test_config_validation_errors():
    bad = {**VALID_CONFIG}
    bad["coordinates"] = [1, 2, 3]
    with pytest.raises(ValueError, match="coordinates"):
        Config.from_dict(bad)

    bad = {**VALID_CONFIG}
    bad["num_chargers"] = 0
    with pytest.raises(ValueError, match="num_chargers"):
        Config.from_dict(bad)

    bad = {**VALID_CONFIG}
    bad["num_chargers"] = 10
    with pytest.raises(ValueError, match="num_chargers cannot exceed"):
        Config.from_dict(bad)

    bad = {**VALID_CONFIG}
    bad["od_demand"] = {"7,26": [60]}
    with pytest.raises(ValueError, match="od_demand"):
        Config.from_dict(bad)


def test_config_to_json_roundtrip():
    cfg = Config.from_dict(VALID_CONFIG)
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        cfg.to_json(f.name)
        with open(f.name, 'r') as rf:
            raw = json.load(rf)
        os.unlink(f.name)
    assert raw["coordinates"] == VALID_CONFIG["coordinates"]
    assert raw["queue_simulation"]["K"] == 8


def test_get_od_demand_tuples():
    cfg = Config.from_dict(VALID_CONFIG)
    od = cfg.get_od_demand_tuples()
    assert (7, 26) in od
    assert od[(7, 26)] == (60, 120)


def test_get_queue_param():
    cfg = Config.from_dict(VALID_CONFIG)
    assert cfg.get_queue_param("K") == 8
    assert cfg.get_queue_param("NONEXISTENT", "default") == "default"
