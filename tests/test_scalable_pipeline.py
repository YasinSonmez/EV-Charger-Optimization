"""Focused contracts for coordinate-driven generated scenarios."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.config import Config
from src.scenario_generation import generate_scenario


def test_generated_config_needs_no_internal_node_ids():
    config = Config.from_dict({
        "name": "generated",
        "coordinates": [39.0, 38.9, -77.0, -77.1],
        "network": {"expected_nodes": 100},
        "scenario_generation": {
            "candidate_count": 5, "num_chargers": 2,
            "od_pair_count": 1, "demand": {"F1": 60, "F2": 120},
        },
        "pipeline": {"bpr_generation": {"mode": "capacity_fraction_strict"}},
        "queue_simulation": {"NUM_ITERS": 20, "MAX_NE_ITERATIONS": 200},
    })
    assert config.possible_charger_positions == []
    assert config.od_demand == {}
    assert config.scenario_generation["enabled"] is True
    assert config.pipeline["bpr_generation"]["allow_proxy"] is False


def test_scenario_generation_is_deterministic_and_feasible():
    nodes = pd.DataFrame({
        "node_id": list(range(8)),
        "lon": [-77.04, -77.03, -77.02, -77.01, -77.01, -77.02, -77.03, -77.04],
        "lat": [38.90, 38.90, 38.90, 38.90, 38.91, 38.91, 38.91, 38.91],
    })
    starts = list(range(8)) + list(range(8))
    ends = [(value + 1) % 8 for value in range(8)] + [(value - 1) % 8 for value in range(8)]
    edges = pd.DataFrame({
        "link_id": list(range(16)), "start_node_id": starts, "end_node_id": ends,
        "length": [100.0] * 16, "travel_time": [10.0] * 16,
        "type": ["secondary"] * 16,
    })
    road_net = SimpleNamespace(nodes=nodes, edges=edges)
    settings = {
        "candidate_count": 3, "interchange_merge_diameter_m": 250,
        "od_pair_count": 1, "boundary_pool_size": 8,
        "demand": {"F1": 2, "F2": 3}, "seed": 42,
    }
    first = generate_scenario(road_net, settings)
    second = generate_scenario(road_net, settings)
    assert first == second
    assert len(first.candidate_node_ids) == 3
    assert list(first.od_demand.values()) == [[2, 3]]


def test_od_corridor_candidates_respect_detour_limit_for_multiple_ods():
    nodes = pd.DataFrame({
        "node_id": list(range(12)),
        "lon": [-77.06 + 0.01 * (i % 6) for i in range(12)],
        "lat": [38.90] * 6 + [38.91] * 6,
    })
    starts, ends = [], []
    undirected = (
        [(i, i + 1) for i in range(5)]
        + [(i, i + 1) for i in range(6, 11)]
        + [(0, 6), (5, 11)]
    )
    for start, end in undirected:
        starts.extend([start, end])
        ends.extend([end, start])
    edges = pd.DataFrame({
        "link_id": list(range(len(starts))),
        "start_node_id": starts, "end_node_id": ends,
        "length": [100.0] * len(starts),
        "travel_time": [10.0] * len(starts),
        "type": ["secondary"] * len(starts),
    })
    road_net = SimpleNamespace(nodes=nodes, edges=edges)
    settings = {
        "candidate_count": 4,
        "candidate_strategy": "od_corridor_interchanges",
        "candidate_max_detour_ratio": 1.25,
        "candidate_corridor_radius_m": 500,
        "interchange_merge_diameter_m": 250,
        "od_pair_count": 2, "boundary_pool_size": 12,
        "demand": {"F1": 2, "F2": 3}, "seed": 42,
    }

    first = generate_scenario(road_net, settings)
    second = generate_scenario(road_net, settings)

    assert first == second
    assert len(first.od_demand) == 2
    assert len(first.candidate_node_ids) == 4
    assert all(
        candidate["minimum_detour_ratio"] <= 1.25
        for candidate in first.metadata["candidates"]
    )
    assert all(
        candidate["minimum_corridor_distance_m"] <= 500
        for candidate in first.metadata["candidates"]
    )
    assert all(
        values["selected_candidates_within_limit"] >= 1
        for values in first.metadata["candidate_diagnostics"]["per_od"].values()
    )
