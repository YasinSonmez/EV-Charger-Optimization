"""Contracts for reviewer-facing placement baselines and route-source guards."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from queue_sim.comparison import _reviewer_baselines
from src.config import Config
from src.contracts import canonical_placement
from src.network_artifact import write_network_artifact


def _write_small_artifact(directory):
    nodes = pd.DataFrame({
        "node_id": range(6),
        "lon": [-77.0 + 0.001 * value for value in range(6)],
        "lat": [38.9 + 0.001 * value for value in range(6)],
    })
    edges = pd.DataFrame([
        {"link_id": 0, "start_node_id": 0, "end_node_id": 1, "travel_time": 1.0},
        {"link_id": 1, "start_node_id": 1, "end_node_id": 2, "travel_time": 1.0},
        {"link_id": 2, "start_node_id": 2, "end_node_id": 5, "travel_time": 1.0},
        {"link_id": 3, "start_node_id": 0, "end_node_id": 3, "travel_time": 1.0},
        {"link_id": 4, "start_node_id": 3, "end_node_id": 4, "travel_time": 1.0},
        {"link_id": 5, "start_node_id": 4, "end_node_id": 5, "travel_time": 1.0},
        {"link_id": 6, "start_node_id": 1, "end_node_id": 4, "travel_time": 1.0},
        {"link_id": 7, "start_node_id": 3, "end_node_id": 2, "travel_time": 1.0},
    ])
    write_network_artifact(nodes, edges, directory)
    return Path(directory)


def _reviewer_fixture(tmp_path):
    artifact_dir = _write_small_artifact(tmp_path / "network")
    experiment_dir = tmp_path / "run"
    experiment_dir.mkdir()
    (experiment_dir / "resolved_config.json").write_text(
        '{"generated_scenario": {"candidates": ['
        '{"node_id": 2, "minimum_detour_ratio": 1.05},'
        '{"node_id": 4, "minimum_detour_ratio": 1.06},'
        '{"node_id": 1, "minimum_detour_ratio": 1.10},'
        '{"node_id": 3, "minimum_detour_ratio": 1.12}]}}'
    )
    pairs = [(a, b) for a in (1, 2, 3, 4) for b in (1, 2, 3, 4) if a < b]
    cg_objectives = {
        placement: float(100 + 10 * index) for index, placement in enumerate(pairs)
    }
    cg_objectives[(1, 3)] = 100.0
    cg_objectives[(1, 2)] = 102.0
    queue_objectives = {
        placement: float(50 + 5 * index) for index, placement in enumerate(pairs)
    }
    queue_objectives[(2, 4)] = 50.0
    queue_objectives[(1, 2)] = 52.0
    cg_search = {
        "greedy": {"placement": [1, 2]},
        "single_swap": {"placement": [1, 4]},
    }
    queue_search = {
        "greedy": {"placement": [2, 3]},
        "single_swap": {"placement": [2, 3]},
    }
    config = SimpleNamespace(
        possible_charger_positions=[1, 2, 3, 4], num_chargers=2,
    )
    result = _reviewer_baselines(
        config, experiment_dir, artifact_dir, cg_objectives, queue_objectives,
        cg_search, queue_search,
    )
    return result, cg_objectives, queue_objectives, pairs


def test_reviewer_baselines_report_regret_and_random_expectation(tmp_path):
    result, cg_objectives, queue_objectives, pairs = _reviewer_fixture(tmp_path)
    expected_means = {
        "congestion_game": sum(cg_objectives.values()) / len(cg_objectives),
        "queue": sum(queue_objectives.values()) / len(queue_objectives),
    }
    for model in ("congestion_game", "queue"):
        strategies = result[model]
        assert set(strategies) == {
            "greedy", "single_swap", "exhaustive", "minimum_detour",
            "weighted_betweenness", "uniform_random_expectation",
        }
        assert strategies["exhaustive"]["regret_pct"] == pytest.approx(0.0)
        expected_random = expected_means[model]
        assert strategies["uniform_random_expectation"]["objective"] == pytest.approx(
            expected_random
        )
        optimum = min(
            cg_objectives.values() if model == "congestion_game"
            else queue_objectives.values()
        )
        assert strategies["uniform_random_expectation"]["regret_pct"] == pytest.approx(
            100.0 * (expected_random - optimum) / optimum
        )
        for method, outcome in strategies.items():
            if method != "uniform_random_expectation":
                assert tuple(outcome["placement"]) == canonical_placement(
                    outcome["placement"]
                )
    queue = result["queue"]
    assert tuple(queue["exhaustive"]["placement"]) == (2, 4)
    assert tuple(queue["minimum_detour"]["placement"]) == (2, 4)
    greedy = queue_objectives[(2, 3)]
    assert queue["greedy"]["regret_pct"] == pytest.approx(
        100.0 * (greedy - 50.0) / 50.0
    )
    assert tuple(result["congestion_game"]["exhaustive"]["placement"]) == (1, 3)


def test_reviewer_baselines_betweenness_is_a_scored_candidate_subset(tmp_path):
    result, _cg_objectives, _queue_objectives, _pairs = _reviewer_fixture(tmp_path)
    betweenness = tuple(result["queue"]["weighted_betweenness"]["placement"])
    assert betweenness in {(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)}
    assert betweenness == tuple(
        result["congestion_game"]["weighted_betweenness"]["placement"]
    )


def test_config_rejects_cg_proportional_initialization_with_independent_routes():
    raw = {
        "coordinates": [38.98211, 38.975, -76.93006, -76.93704],
        "num_chargers": 2,
        "possible_charger_positions": [14, 20, 21],
        "od_demand": {"7,26": [60, 120]},
        "max_iter": 1000,
        "single_swap": True,
        "calculate_on_all_possible_positions": True,
        "queue_simulation": {
            "K": 8, "NUM_ITERS": 3, "N": 5,
            "route_source": "independent_network_routes",
            "initialization": "cg_proportional",
        },
    }
    with pytest.raises(ValueError, match="cg_proportional requires CG-recovered"):
        Config.from_dict(raw)
    raw["queue_simulation"]["initialization"] = "uniform"
    assert Config.from_dict(raw).queue_simulation["initialization"] == "uniform"
