"""Offline output-contract tests for the complete experiment artifact set."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd

from src.network_artifact import write_network_artifact
from src.sanity_checks import validate_experiment_outputs


def _write_fixture(root: Path) -> str:
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1, 2, 3],
            "lon": [0.0, 1.0, 2.0, 3.0],
            "lat": [0.0, 0.0, 0.0, 0.0],
        }
    )
    edges = pd.DataFrame(
        {
            "link_id": [0, 1, 2, 3],
            "start_node_id": [0, 1, 2, 3],
            "end_node_id": [1, 2, 3, 0],
            "edge_key": ["a", "b", "c", "d"],
            "length": [1.0, 1.0, 1.0, 1.0],
        }
    )
    manifest = write_network_artifact(
        nodes,
        edges,
        root / "network",
        stage_counts={"raw": {"nodes": 4, "edges": 4}, "final_scc": {"nodes": 4, "edges": 4}},
    )
    (root / "network_manifest.json").write_text(json.dumps(manifest))

    for relative in (
        "run_config.json",
        "run_manifest.json",
        "experiment_summary.json",
        "report.md",
        "run_summary.txt",
        "plots/pruning_phases.png",
    ):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}" if path.suffix == ".json" else "fixture")

    demand = {"0,2": [2, 1], "1,3": [1, 2]}
    link_flows = {
        link_id: {
            "start_node_id": int(edges.loc[link_id, "start_node_id"]),
            "end_node_id": int(edges.loc[link_id, "end_node_id"]),
            "total_flow": 3.0,
            "non_charging_flow": 1.0,
            "charging_flows": {},
        }
        for link_id in edges["link_id"]
    }
    routes = [
        {"route_id": "r0", "origin": 0, "destination": 2, "type": "non_charging", "flow": 2.0, "link_ids": [0, 1], "links": [0, 1]},
        {"route_id": "r1", "origin": 0, "destination": 2, "type": "charging", "charger": 1, "flow": 1.0, "link_ids": [0, 1], "links": [0, 1]},
        {"route_id": "r2", "origin": 1, "destination": 3, "type": "non_charging", "flow": 1.0, "link_ids": [1, 2], "links": [1, 2]},
        {"route_id": "r3", "origin": 1, "destination": 3, "type": "charging", "charger": 2, "flow": 2.0, "link_ids": [1, 2], "links": [1, 2]},
    ]
    all_opt = {
        "network_hash": manifest["network_hash"],
        "run_configuration": {"od_demand": demand, "network_hash": manifest["network_hash"]},
        "network_link_connectivity": [
            {"link_id": int(row.link_id), "start_node_id": int(row.start_node_id), "end_node_id": int(row.end_node_id)}
            for row in edges.itertuples()
        ],
        "configurations": {
            frozenset({2}): {
                "objective_value": 12.5,
                "link_flows": link_flows,
                "reconstruction_results": {"k_metrics": {16: {"routes": routes}}},
            }
        },
    }
    with (root / "all_optimization_results.pkl").open("wb") as handle:
        pickle.dump(all_opt, handle)

    (root / "run_manifest.json").write_text(json.dumps({"provenance": {"network_hash": manifest["network_hash"]}}))
    (root / "experiment_summary.json").write_text(json.dumps({"provenance": {"network_hash": manifest["network_hash"]}}))

    queue = root / "queue"
    queue.mkdir()
    placement = {
        "network_hash": manifest["network_hash"],
        "status": "ok",
        "assignments": {
            "F1": {(0, 2): [2], (1, 3): [1]},
            "F2": {(0, 2): [1], (1, 3): [2]},
        },
    }
    with (queue / "NE_path_assignments.pkl").open("wb") as handle:
        pickle.dump({"2": placement}, handle)
    (queue / "queue_manifest.json").write_text(
        json.dumps({"network_hash": manifest["network_hash"], "configuration_count": 1, "failed_configurations": {}})
    )
    (queue / "comparison_results.json").write_text(
        json.dumps(
            {
                "network_hash": manifest["network_hash"],
                "best_greedy": {"positions": [2], "avg_travel_time": 10.0},
                "best_exhaustive": {"positions": [2], "avg_travel_time": 10.0},
                "suboptimality_pct": 0.0,
            }
        )
    )
    (root / "run_config.json").write_text(json.dumps({"od_demand": demand}))
    return manifest["network_hash"]


def test_two_od_full_artifact_contract_passes(tmp_path):
    network_hash = _write_fixture(tmp_path)
    result = validate_experiment_outputs(tmp_path, require_queue=True)

    assert result["valid"], result["errors"]
    assert result["network_hash"] == network_hash
    assert result["optimization"]["configuration_count"] == 1
    assert result["optimization"]["demand_class_count"] == 4
    assert result["optimization"]["route_count"] == 4
    assert result["queue"]["configuration_count"] == 1


def test_output_validator_detects_network_hash_drift(tmp_path):
    _write_fixture(tmp_path)
    all_opt_path = tmp_path / "all_optimization_results.pkl"
    with all_opt_path.open("rb") as handle:
        data = pickle.load(handle)
    data["network_hash"] = "wrong"
    with all_opt_path.open("wb") as handle:
        pickle.dump(data, handle)

    result = validate_experiment_outputs(tmp_path, require_queue=True)

    assert not result["valid"]
    assert any("network_hash" in error for error in result["errors"])


def test_output_validator_detects_unknown_route_link(tmp_path):
    _write_fixture(tmp_path)
    all_opt_path = tmp_path / "all_optimization_results.pkl"
    with all_opt_path.open("rb") as handle:
        data = pickle.load(handle)
    route = data["configurations"][frozenset({2})]["reconstruction_results"]["k_metrics"][16]["routes"][0]
    route["link_ids"] = [999]
    with all_opt_path.open("wb") as handle:
        pickle.dump(data, handle)

    result = validate_experiment_outputs(tmp_path)

    assert not result["valid"]
    assert any("unknown link IDs" in error for error in result["errors"])


def test_output_validator_accepts_configuration_specific_charger_self_link(tmp_path):
    _write_fixture(tmp_path)
    all_opt_path = tmp_path / "all_optimization_results.pkl"
    with all_opt_path.open("rb") as handle:
        data = pickle.load(handle)

    config = data["configurations"][frozenset({2})]
    config["link_flows"][4] = {
        "start_node_id": 2,
        "end_node_id": 2,
        "total_flow": 1.0,
        "non_charging_flow": 0.0,
        "charging_flows": {2: 1.0},
    }
    config["link_connectivity"] = [
        {
            "link_id": int(link_id),
            "start_node_id": int(flow["start_node_id"]),
            "end_node_id": int(flow["end_node_id"]),
        }
        for link_id, flow in config["link_flows"].items()
    ]
    with all_opt_path.open("wb") as handle:
        pickle.dump(data, handle)

    result = validate_experiment_outputs(tmp_path)
    assert result["valid"], result["errors"]
