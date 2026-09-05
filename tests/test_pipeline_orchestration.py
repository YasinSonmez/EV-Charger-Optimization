"""Deterministic orchestration test without OSM, Overpass, or liblsp."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline
from src.network_artifact import write_network_artifact
from src.sanity_checks import validate_experiment_outputs


class _FakeRoadNet:
    def __init__(self, name):
        self.name = name
        self.nodes = pd.DataFrame(
            {"node_id": [0, 1, 2, 3], "lon": [0.0, 1.0, 2.0, 3.0], "lat": [0.0] * 4}
        )
        self.edges = pd.DataFrame(
            {
                "link_id": [0, 1, 2, 3],
                "start_node_id": [0, 1, 2, 3],
                "end_node_id": [1, 2, 3, 0],
                "edge_key": ["a", "b", "c", "d"],
                "length": [1.0] * 4,
            }
        )
        self.stage_counts = {
            "raw": {"nodes": 4, "edges": 4},
            "final_scc": {"nodes": 4, "edges": 4},
        }
        self.stage_maps = {}
        self.solver_metadata = {"solver": "fixture", "status": "optimal"}

    def get_map(self, *args, **kwargs):
        return None

    def export_artifact(self, output_dir, source=None):
        return write_network_artifact(
            self.nodes,
            self.edges,
            output_dir,
            stage_counts=self.stage_counts,
            stage_maps=self.stage_maps,
            source=source,
        )


class _FakeGrid:
    def __init__(self, road_net):
        self.chargers = [2]
        self.travel_time_obj = 12.5
        self.net = road_net


def _write_fake_cg_results(output_dir, road_net, od_demand):
    link_flows = {
        int(row.link_id): {
            "start_node_id": int(row.start_node_id),
            "end_node_id": int(row.end_node_id),
            "total_flow": 3.0,
            "non_charging_flow": 1.0,
            "charging_flows": {},
        }
        for row in road_net.edges.itertuples()
    }
    routes = [
        {"route_id": "0_2_f1", "origin": 0, "destination": 2, "type": "non_charging", "flow": 2.0, "link_ids": [0, 1], "links": [0, 1]},
        {"route_id": "0_2_f2", "origin": 0, "destination": 2, "type": "charging", "charger": 2, "flow": 1.0, "link_ids": [0, 1], "links": [0, 1]},
        {"route_id": "1_3_f1", "origin": 1, "destination": 3, "type": "non_charging", "flow": 1.0, "link_ids": [1, 2], "links": [1, 2]},
        {"route_id": "1_3_f2", "origin": 1, "destination": 3, "type": "charging", "charger": 2, "flow": 2.0, "link_ids": [1, 2], "links": [1, 2]},
    ]
    result = {
        "run_configuration": {"od_demand": od_demand},
        "configurations": {
            frozenset({2}): {
                "objective_value": 12.5,
                "link_flows": link_flows,
                "reconstruction_results": {"k_metrics": {16: {"routes": routes}}},
            }
        },
        "network_link_connectivity": [
            {"link_id": int(row.link_id), "start_node_id": int(row.start_node_id), "end_node_id": int(row.end_node_id)}
            for row in road_net.edges.itertuples()
        ],
    }
    with (Path(output_dir) / "all_optimization_results.pkl").open("wb") as handle:
        pickle.dump(result, handle)


def _fake_bpr(*args, **kwargs):
    return (
        pd.DataFrame(
            {
                "link_id": [0, 1, 2, 3],
                "x_vector": [np.array([0.0, 1.0])] * 4,
                "y_vector": [np.array([1.0, 1.1])] * 4,
            }
        ),
        object(),
    )


def _fake_plot(_data, path, **kwargs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(b"fixture")


def test_pipeline_orchestrator_produces_valid_two_od_outputs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("src.road_network.RoadNet", _FakeRoadNet)
    monkeypatch.setattr(pipeline, "load_or_fit_model", _fake_bpr)
    monkeypatch.setattr(pipeline, "_plot_pruning_phases", _fake_plot)
    monkeypatch.setattr(pipeline, "_plot_bpr_fit_samples", _fake_plot)
    monkeypatch.setattr(pipeline, "_plot_timing_breakdown", _fake_plot)

    def fake_outer(**kwargs):
        grid = _FakeGrid(kwargs["road_net"])
        _write_fake_cg_results(kwargs["output_dir"], kwargs["road_net"], kwargs["od_demand"])
        return [grid], [], kwargs["output_dir"]

    monkeypatch.setattr(pipeline, "outer_optimization", fake_outer)

    def fake_nash(config, experiment_dir, all_opt_results_path, artifact_dir=None, seed_manager=None, **kwargs):
        with open(all_opt_results_path, "rb") as handle:
            all_opt = pickle.load(handle)
        network_hash = json.loads((Path(artifact_dir) / "network_manifest.json").read_text())["network_hash"]
        queue_dir = Path(experiment_dir) / "queue"
        queue_dir.mkdir(parents=True, exist_ok=True)
        assignments = {"F1": {(0, 2): [2], (1, 3): [1]}, "F2": {(0, 2): [1], (1, 3): [2]}}
        with (queue_dir / "NE_path_assignments.pkl").open("wb") as handle:
            pickle.dump({"2": {"network_hash": network_hash, "status": "ok", "assignments": assignments}}, handle)
        (queue_dir / "queue_manifest.json").write_text(
            json.dumps({"network_hash": network_hash, "configuration_count": 1, "failed_configurations": {}})
        )
        return str(queue_dir / "NE_path_assignments.pkl"), {"2": [0.0]}

    def fake_comparison(config, experiment_dir, all_opt_results_path, ne_assignments_path, artifact_dir=None, seed_manager=None, **kwargs):
        network_hash = json.loads((Path(artifact_dir) / "network_manifest.json").read_text())["network_hash"]
        results = {
            "network_hash": network_hash,
            "best_greedy": {"positions": [2], "avg_travel_time": 10.0},
            "best_exhaustive": {"positions": [2], "avg_travel_time": 10.0},
            "suboptimality_pct": 0.0,
            "greedy_results": [{"positions": [2], "avg_travel_time": 10.0}],
            "exhaustive_results": [{"positions": [2], "avg_travel_time": 10.0}],
            "config": {"N": 1, "K": 16, "single_swap": False},
        }
        (Path(experiment_dir) / "queue" / "comparison_results.json").write_text(json.dumps(results))
        return results

    monkeypatch.setattr("queue_sim.find_nash.find_nash_assignments", fake_nash)
    monkeypatch.setattr("queue_sim.comparison.run_comparison", fake_comparison)
    monkeypatch.setattr(pipeline, "QUEUE_SIM_AVAILABLE", True)

    config_path = tmp_path / "fixture_config.json"
    config_path.write_text(
        json.dumps(
            {
                "coordinates": [1.0, 0.0, 1.0, 0.0],
                "num_chargers": 1,
                "possible_charger_positions": [2],
                "od_demand": {"0,2": [2, 1], "1,3": [1, 2]},
                "use_cvxpy": False,
                "calculate_on_all_possible_positions": True,
                "route_analysis": {"k_values": [16]},
                "queue_simulation": {"enabled": True, "K": 16, "N": 1, "single_swap": False},
                "pipeline": {"random_seed": 7, "bpr_generation": {"min_samples": 2}},
            }
        )
    )

    experiment_dir = Path(pipeline.run_pipeline(str(config_path)))
    result = validate_experiment_outputs(experiment_dir, require_queue=True)

    assert result["valid"], result["errors"]
    assert (experiment_dir / "all_optimization_results.pkl").exists()
    assert (experiment_dir / "queue" / "comparison_results.json").exists()
    sanity = json.loads((experiment_dir / "sanity_check.json").read_text())
    assert sanity["valid"] is True
    inventory = json.loads((experiment_dir / "artifact_inventory.json").read_text())
    assert inventory["full_scan"] is False
    assert inventory["scan_status"] == "skipped"
    assert inventory["total_bytes"] is None
    status = json.loads((experiment_dir / "status.json").read_text())
    assert status["artifact_bytes"] is None
