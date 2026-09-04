import json
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString

from src.config import NetworkConfig
from src.graph_cache import _cache_key
from src.network_coarsening import (
    contract_freeflow_shortcut_skeleton,
    select_charger_candidate_layer,
)
from src.network_pruning import (
    ROAD_PROFILES,
    consolidate_intersections,
    filter_highways,
    path_fidelity,
    prepare_source_graph,
    project_graph,
    recover_directed_connectors,
    removable_pass_through_nodes,
    topology_simplify,
)


def _graph(crs="EPSG:32618"):
    graph = nx.MultiDiGraph(crs=crs)
    return graph


def _add_node(graph, node, x, y):
    graph.add_node(node, x=float(x), y=float(y))


def _add_edge(graph, u, v, *, length=10.0, highway="primary", key=None):
    geometry = LineString([
        (graph.nodes[u]["x"], graph.nodes[u]["y"]),
        (graph.nodes[v]["x"], graph.nodes[v]["y"]),
    ])
    graph.add_edge(
        u,
        v,
        key=key,
        length=float(length),
        highway=highway,
        maxspeed="50 kph",
        lanes="2",
        geometry=geometry,
    )


def test_mixed_highway_tag_keeps_any_allowed_member():
    graph = _graph()
    _add_node(graph, 0, 0, 0)
    _add_node(graph, 1, 10, 0)
    _add_node(graph, 2, 20, 0)
    _add_edge(graph, 0, 1, highway=["residential", "primary_link"])
    _add_edge(graph, 1, 2, highway="residential")
    prepared = prepare_source_graph(graph)

    result = filter_highways(prepared, ROAD_PROFILES["primary_plus"])

    assert result.has_edge(0, 1)
    assert not result.has_edge(1, 2)


def test_topology_simplification_preserves_path_cost_geometry_and_provenance():
    graph = _graph()
    for node, x in enumerate((0, 10, 30)):
        _add_node(graph, node, x, 0)
    _add_edge(graph, 0, 1, length=10)
    _add_edge(graph, 1, 2, length=20)
    _add_edge(graph, 2, 1, length=20)
    _add_edge(graph, 1, 0, length=10)
    prepared = prepare_source_graph(graph)

    result = topology_simplify(prepared)

    assert 1 not in result
    assert result.has_edge(0, 2)
    edge = next(iter(result.get_edge_data(0, 2).values()))
    assert edge["length"] == pytest.approx(30)
    assert len(edge["source_edge_ids"]) == 2
    assert edge["travel_time"] == pytest.approx(
        prepared.edges[0, 1, 0]["travel_time"] + prepared.edges[1, 2, 0]["travel_time"]
    )
    assert len(edge["geometry"].coords) == 3
    assert removable_pass_through_nodes(result) == []


def test_parallel_edges_are_not_dropped_or_cross_multiplied():
    graph = _graph()
    for node, x in enumerate((0, 10, 20)):
        _add_node(graph, node, x, 0)
    _add_edge(graph, 0, 1, key=0)
    _add_edge(graph, 0, 1, key=1)
    _add_edge(graph, 1, 2, key=0)
    _add_edge(graph, 1, 2, key=1)
    prepared = prepare_source_graph(graph)

    result = topology_simplify(prepared)

    assert 1 in result
    assert result.number_of_edges(0, 1) == 2
    assert result.number_of_edges(1, 2) == 2
    assert not result.has_edge(0, 2)
    assert removable_pass_through_nodes(result) == []


def test_one_way_chain_is_safely_collapsed():
    graph = _graph()
    for node, x in enumerate((0, 10, 30)):
        _add_node(graph, node, x, 0)
    _add_edge(graph, 0, 1, length=10)
    _add_edge(graph, 1, 2, length=20)

    result = topology_simplify(prepare_source_graph(graph))

    assert set(result) == {0, 2}
    edge = next(iter(result.get_edge_data(0, 2).values()))
    assert edge["length"] == pytest.approx(30)
    assert len(edge["source_edge_ids"]) == 2


def test_strongly_connected_consolidation_does_not_merge_one_way_cluster():
    graph = _graph()
    for node, x in enumerate((0, 5, 10)):
        _add_node(graph, node, x, 0)
    _add_edge(graph, 0, 1)
    _add_edge(graph, 1, 2)
    _add_edge(graph, 0, 2)
    prepared = prepare_source_graph(graph)

    consolidated, mapping, clusters = consolidate_intersections(
        prepared,
        10,
        require_induced_strong_connectivity=True,
    )

    assert not clusters
    assert not any(str(node).startswith("_J") for node in consolidated)
    assert mapping[0] == 0
    assert mapping[2] == 2


def test_candidate_layer_is_deterministic_and_bounded():
    graph = _graph()
    for node, x in enumerate((0, 100, 200, 300, 400)):
        _add_node(graph, node, x, 0)
    for node in range(4):
        _add_edge(graph, node, node + 1)
        _add_edge(graph, node + 1, node)
    graph.edges[1, 2, 0]["highway"] = "motorway_link"
    graph.edges[2, 1, 0]["highway"] = "motorway_link"

    first, metadata, diagnostics = select_charger_candidate_layer(
        graph, max_candidates=3, interchange_merge_diameter_m=250
    )
    second, _, _ = select_charger_candidate_layer(
        graph, max_candidates=3, interchange_merge_diameter_m=250
    )

    assert first == second
    assert len(first) == 3
    assert diagnostics["candidate_count"] == 3
    assert diagnostics["interchange_candidates"] == 1
    assert all(node in graph for node in metadata)


def test_candidate_layer_caps_many_interchange_groups():
    graph = _graph()
    for node, x in enumerate((0, 10, 1000, 1010, 2000, 2010)):
        _add_node(graph, node, x, 0)
    for start in (0, 2, 4):
        _add_edge(graph, start, start + 1, highway="motorway_link")
        _add_edge(graph, start + 1, start, highway="motorway_link")

    candidates, _, diagnostics = select_charger_candidate_layer(
        graph, max_candidates=2, interchange_merge_diameter_m=100
    )

    assert len(candidates) == 2
    assert diagnostics["candidate_count"] == 2
    assert diagnostics["interchange_groups"] == 3


def test_freeflow_shortcut_skeleton_preserves_survivor_distances():
    graph = _graph()
    for node, x in enumerate((0, 10, 20, 30)):
        _add_node(graph, node, x, 0)
    for node in range(3):
        _add_edge(graph, node, node + 1, length=10)
        _add_edge(graph, node + 1, node, length=10)
    _add_edge(graph, 3, 0, length=10)
    _add_edge(graph, 0, 3, length=10)
    prepared = prepare_source_graph(graph)

    skeleton, diagnostics = contract_freeflow_shortcut_skeleton(
        prepared, target_nodes=3
    )
    fidelity = path_fidelity(
        prepared, skeleton, {node: node for node in skeleton}, seed=3
    )

    assert len(skeleton) == 3
    assert fidelity["reachability_preserved_fraction"] == 1.0
    assert fidelity["path_distortion_max"] == pytest.approx(0.0)
    assert diagnostics["congestion_compatible"] is False
    assert diagnostics["max_source_edges_per_shortcut"] >= 2


def test_simplification_uses_bottleneck_lane_count():
    graph = _graph()
    for node, x in enumerate((0, 10, 20)):
        _add_node(graph, node, x, 0)
    _add_edge(graph, 0, 1)
    _add_edge(graph, 1, 2)
    graph.edges[0, 1, 0]["lanes"] = "3"
    graph.edges[1, 2, 0]["lanes"] = "1"

    result = topology_simplify(prepare_source_graph(graph))

    edge = next(iter(result.get_edge_data(0, 2).values()))
    assert edge["lanes_numeric"] == 1


def test_consolidation_does_not_join_nearby_disconnected_roads():
    graph = _graph()
    # Two close triangles are geometrically overlapping at a 10 m radius but
    # belong to separate weak components and must remain separate clusters.
    for node, xy in {
        0: (0, 0), 1: (8, 0), 2: (4, 7),
        3: (10, 0), 4: (18, 0), 5: (14, 7),
    }.items():
        _add_node(graph, node, *xy)
    for cycle in ((0, 1, 2), (3, 4, 5)):
        for u, v in zip(cycle, cycle[1:] + cycle[:1]):
            _add_edge(graph, u, v)
            _add_edge(graph, v, u)
    prepared = prepare_source_graph(graph)
    prepared.graph["simplified"] = True

    result, mapping, _ = consolidate_intersections(prepared, 10)

    assert mapping[0] != mapping[3]
    assert nx.number_weakly_connected_components(result) == 2


def test_identity_consolidation_has_exact_path_fidelity():
    graph = _graph()
    for node, x in enumerate((0, 10, 20, 30)):
        _add_node(graph, node, x, 0)
    for u, v in ((0, 1), (1, 2), (2, 3)):
        _add_edge(graph, u, v)
        _add_edge(graph, v, u)
    prepared = prepare_source_graph(graph)
    prepared.graph["simplified"] = True

    result, mapping, _ = consolidate_intersections(prepared, 0)
    fidelity = path_fidelity(prepared, result, mapping)

    assert fidelity["reachability_preserved_fraction"] == 1.0
    assert fidelity["introduced_reachable_pairs"] == 0
    assert fidelity["path_distortion_max"] == 0.0


def test_connector_recovery_adds_missing_reverse_direction():
    narrow = _graph()
    for node, xy in {0: (0, 0), 1: (10, 0), 2: (20, 0), 3: (30, 0), 4: (15, 10)}.items():
        _add_node(narrow, node, *xy)
    for u, v in ((0, 1), (1, 0), (2, 3), (3, 2), (1, 2)):
        _add_edge(narrow, u, v, highway="primary")
    broad = narrow.copy()
    _add_edge(broad, 3, 4, highway="secondary")
    _add_edge(broad, 4, 0, highway="secondary")
    narrow = prepare_source_graph(narrow)
    broad = prepare_source_graph(broad)

    recovered, metadata = recover_directed_connectors(narrow, broad, target_ratio=0.9)

    assert metadata["final_scc_wcc_ratio"] == 1.0
    assert metadata["edges_added"] == 2
    assert nx.is_strongly_connected(recovered)


def test_cache_key_changes_when_simplification_policy_changes():
    kwargs = {
        "network_type": "drive",
        "retain_all": True,
        "truncate_by_edge": False,
        "custom_filter": None,
    }
    simplified = _cache_key((0, 0, 1, 1), ["primary"], simplify=True, **kwargs)
    unsimplified = _cache_key((0, 0, 1, 1), ["primary"], simplify=False, **kwargs)
    assert simplified != unsimplified


def test_projection_round_trip_does_not_require_geopandas_graph_conversion():
    graph = _graph(crs="EPSG:4326")
    _add_node(graph, 0, -77.0, 38.9)
    _add_node(graph, 1, -76.99, 38.91)
    _add_edge(graph, 0, 1, length=1400)
    prepared = prepare_source_graph(graph)

    projected = project_graph(prepared)
    restored = project_graph(projected, to_crs="EPSG:4326")

    assert projected.graph["crs"].is_projected
    assert restored.nodes[0]["x"] == pytest.approx(-77.0, abs=1e-7)
    assert restored.nodes[0]["y"] == pytest.approx(38.9, abs=1e-7)
    assert restored.edges[0, 1, 0]["length"] == 1400


def test_network_only_config_requires_no_charger_or_demand_fields(tmp_path):
    path = tmp_path / "network.json"
    path.write_text(json.dumps({
        "name": "minimal",
        "coordinates": [1.0, 0.0, 1.0, 0.0],
        "road_filter": {
            "sweep_profiles": ["primary_plus"],
            "sweep_radii_m": [0, 5],
        },
    }))

    config = NetworkConfig.from_json(path)

    assert config.name == "minimal"
    assert config.coordinates == [1.0, 0.0, 1.0, 0.0]


def test_pruning_sweep_writes_deterministic_report_bundle(tmp_path, monkeypatch):
    graph = _graph()
    coordinates = {
        0: (0, 0), 1: (100, 0), 2: (0, 100), 3: (-100, 0), 4: (0, -100),
    }
    for node, xy in coordinates.items():
        _add_node(graph, node, *xy)
    for leaf in (1, 2, 3, 4):
        _add_edge(graph, 0, leaf)
        _add_edge(graph, leaf, 0)

    import src.pruning_study as pruning_study
    monkeypatch.setattr(pruning_study, "get_graph", lambda *args, **kwargs: graph.copy())
    config_path = tmp_path / "sweep.json"
    config_path.write_text(json.dumps({
        "name": "fixture",
        "coordinates": [1.0, 0.0, 1.0, 0.0],
        "output_dir": str(tmp_path / "out"),
        "road_filter": {
            "sweep_profiles": ["primary_plus"],
            "sweep_radii_m": [0, 5],
            "connector_threshold": 0.9,
            "diagnostic_seed": 7,
            "diagnostic_profile": "primary_plus",
            "diagnostic_radius_m": 0,
            "diagnostic_connector_recovery": False,
        },
    }))

    output = pruning_study.run_pruning_sweep(str(config_path))

    output_path = Path(output)
    assert (output_path / "pruning_summary.csv").exists()
    assert (output_path / "pruning_report.md").exists()
    assert (output_path / "run_manifest.json").exists()
    assert (output_path / "selected_network" / "network_manifest.json").exists()
    run_manifest = json.loads((output_path / "run_manifest.json").read_text())
    assert run_manifest["selection_mode"].startswith("explicit configuration override")
    assert run_manifest["leading_variant"]["profile"] == "primary_plus"
    assert run_manifest["leading_variant"]["intersection_radius_m"] == 0
    summary = pd.read_csv(output_path / "pruning_summary.csv")
    assert {
        "source_nodes", "filtered_nodes", "topology_nodes", "consolidated_nodes",
        "cluster_sizes", "cluster_diameters_m", "cluster_external_branch_counts",
        "adjacent_close_pairs_le_20m", "topology_fidelity_valid",
    }.issubset(summary.columns)
    assert {path.name for path in (output_path / "plots").iterdir()} == {
        "road_profile_comparison.png",
        "consolidation_sweep.png",
        "close_node_diagnostics.png",
        "component_comparison.png",
        "pruning_steps.png",
    }
