"""Route-library tests for multi-OD and multi-vehicle coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils import analyze_route_reconstruction
from src.contracts import DemandClass, SeedManager
from queue_sim.route_library import (
    balanced_charger_routes, independent_flow_data, initialize_counts,
)


class _ReconstructionFixture:
    def reconstruct_route_flows(self, *args, **kwargs):
        return {
            (0, 2): {
                "non_charging": [{"path": [0, 1], "link_ids": [0], "flow": 2.0}],
                "charging": {4: [{"path": [0, 1], "link_ids": [0, 1], "flow": 1.0}]},
            },
            (1, 3): {
                "non_charging": [{"path": [1, 2], "link_ids": [1], "flow": 1.0}],
                "charging": {4: [{"path": [1, 2], "link_ids": [1, 2], "flow": 2.0}]},
            },
        }

    def _path_to_link_ids(self, path):
        return list(path)


def test_top_k_route_library_preserves_every_od_and_vehicle_class():
    link_flows = {
        link_id: {"total_flow": 1.0, "start_node_id": 0, "end_node_id": 1}
        for link_id in range(3)
    }

    result = analyze_route_reconstruction(_ReconstructionFixture(), link_flows, k_values=[1])
    routes = result["k_metrics"][1]["routes"]
    groups = {(route["origin"], route["destination"], route["type"]) for route in routes}

    assert groups == {
        (0, 2, "non_charging"),
        (0, 2, "charging"),
        (1, 3, "non_charging"),
        (1, 3, "charging"),
    }
    assert all(np.isfinite(route["flow"]) for route in routes)


def test_balanced_quota_uses_next_best_route_for_remainder():
    routes = {
        30: [{"score": 1}, {"score": 2}, {"score": 4}],
        20: [{"score": 1}, {"score": 2}, {"score": 3}],
        10: [{"score": 1}, {"score": 2}, {"score": 5}],
    }
    selected = balanced_charger_routes(routes, 7, lambda route: route["score"])
    counts = {charger: 0 for charger in routes}
    for charger, candidates in routes.items():
        counts[charger] = sum(route in selected for route in candidates)
    assert counts == {10: 2, 20: 3, 30: 2}


def test_balanced_quota_caps_scarce_chargers_without_failing():
    routes = {
        10: [{"score": 1}],
        20: [{"score": 1}, {"score": 2}, {"score": 3}, {"score": 4}, {"score": 5}],
    }
    selected = balanced_charger_routes(routes, 4, lambda route: route["score"])
    counts = {charger: sum(route in selected for route in candidates)
              for charger, candidates in routes.items()}
    assert counts == {10: 1, 20: 3}
    assert len(selected) == 4


def test_balanced_quota_fails_only_when_no_charger_has_routes():
    with pytest.raises(ValueError, match="No feasible charging routes"):
        balanced_charger_routes({10: [], 20: []}, 4, lambda route: 0)
    single = balanced_charger_routes({10: [{"score": 1}], 20: []}, 2, lambda route: 0)
    assert len(single) == 1


def test_independent_routes_are_balanced_for_multiple_ods():
    edges = []
    link_id = 0
    for origin, destination in ((0, 5), (6, 11)):
        for charger in (2, 3):
            for branch in range(2):
                before = 1000 + origin * 100 + charger * 10 + branch
                after = before + 50
                edges.extend([
                    {"link_id": link_id, "start_node_id": origin,
                     "end_node_id": before, "travel_time": 1 + branch / 10},
                    {"link_id": link_id + 1, "start_node_id": before,
                     "end_node_id": charger, "travel_time": 1},
                    {"link_id": link_id + 2, "start_node_id": charger,
                     "end_node_id": after, "travel_time": 1 + branch / 10},
                    {"link_id": link_id + 3, "start_node_id": after,
                     "end_node_id": destination, "travel_time": 1},
                ])
                link_id += 4
        edges.append({"link_id": link_id, "start_node_id": origin,
                      "end_node_id": destination, "travel_time": 1})
        link_id += 1
    demand = [
        DemandClass(f"{o}_{d}_{kind}", o, d, kind, 4)
        for o, d in ((0, 5), (6, 11)) for kind in ("F1", "F2")
    ]
    result = independent_flow_data(pd.DataFrame(edges), demand, (2, 3), 4)
    assert set(result) == {(0, 5), (6, 11)}
    for group in result.values():
        assert len(group["charging type"]) == 4
        assert sorted(
            sum(route["station node"] == charger for route in group["charging type"])
            for charger in (2, 3)
        ) == [2, 2]


def test_initialization_preserves_demand_and_seed():
    routes = [{"route_id": str(index), "free_flow_time": index + 1} for index in range(4)]
    assert initialize_counts(routes, 10, "uniform") == [3, 3, 2, 2]
    first = initialize_counts(routes, 100, "seeded_random", SeedManager(42), (1, 2))
    second = initialize_counts(routes, 100, "seeded_random", SeedManager(42), (1, 2))
    assert first == second
    assert sum(first) == 100


def test_charging_routes_match_cg_concatenation_semantics():
    """Charging routes are concatenations of simple O-C and C-D legs, matching
    the CG model even when every short O-C leg passes through the destination;
    the shortest such route may legitimately revisit the destination."""
    link_id = 0
    edges = []
    for node in range(1, 20):
        edges.extend([
            {"link_id": link_id, "start_node_id": 0, "end_node_id": node, "travel_time": 0.5},
            {"link_id": link_id + 1, "start_node_id": node, "end_node_id": 30, "travel_time": 0.5},
        ])
        link_id += 2
    edges.extend([
        {"link_id": link_id, "start_node_id": 30, "end_node_id": 31, "travel_time": 1.0},
        {"link_id": link_id + 1, "start_node_id": 0, "end_node_id": 31, "travel_time": 5.0},
        {"link_id": link_id + 2, "start_node_id": 31, "end_node_id": 30, "travel_time": 1.0},
    ])
    demand = [DemandClass("0_30_F1", 0, 30, "F1", 4), DemandClass("0_30_F2", 0, 30, "F2", 4)]
    result = independent_flow_data(pd.DataFrame(edges), demand, (31,), 4)
    group = result[(0, 30)]
    assert len(group["charging type"]) == 4
    for route in group["charging type"]:
        assert route["path"].count(31) == 1
        assert route["path"].count(30) <= 2
        assert route["path"][0] == 0 and route["path"][-1] == 30
    times = [route["free_flow_time"] for route in group["charging type"]]
    assert times == sorted(times)
    assert times[0] == pytest.approx(3.0)
