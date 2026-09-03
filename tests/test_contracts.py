"""Regression tests for canonical pipeline contracts."""

import pandas as pd
import pytest

from src.contracts import SeedManager, normalize_od_demand
from src.network_artifact import load_network_artifact, write_network_artifact


def _network_frames():
    nodes = pd.DataFrame([
        {'node_id': 0, 'lon': 0.0, 'lat': 0.0, 'node_osmid': 10, 'type': 'real'},
        {'node_id': 1, 'lon': 1.0, 'lat': 0.0, 'node_osmid': 20, 'type': 'real'},
    ])
    edges = pd.DataFrame([
        {'link_id': 0, 'start_node_id': 0, 'end_node_id': 1, 'edge_key': 'slow', 'length': 2.0},
        {'link_id': 1, 'start_node_id': 0, 'end_node_id': 1, 'edge_key': 'fast', 'length': 1.0},
        {'link_id': 2, 'start_node_id': 1, 'end_node_id': 0, 'edge_key': 'return', 'length': 1.0},
    ])
    return nodes, edges


def test_normalize_multiple_od_legacy_and_typed_inputs():
    records = normalize_od_demand({
        '0,1': [3, 2],
        (1, 0): {'F1': 4, 'F2': 1},
    })
    assert [(r.origin, r.destination, r.vehicle_type, r.demand) for r in records] == [
        (0, 1, 'F1', 3), (0, 1, 'F2', 2),
        (1, 0, 'F1', 4), (1, 0, 'F2', 1),
    ]


def test_canonical_artifact_preserves_parallel_edges_and_hash(tmp_path):
    nodes, edges = _network_frames()
    first = write_network_artifact(nodes, edges, tmp_path / 'first', source={'seed': 42})
    reversed_manifest = write_network_artifact(
        nodes.iloc[::-1], edges.iloc[::-1], tmp_path / 'second', source={'seed': 42}
    )
    assert first['network_hash'] == reversed_manifest['network_hash']
    loaded_nodes, loaded_edges, loaded_manifest = load_network_artifact(tmp_path / 'first')
    assert len(loaded_edges) == 3
    assert loaded_edges[['start_node_id', 'end_node_id']].duplicated().any()
    assert set(loaded_edges['edge_key']) == {'slow', 'fast', 'return'}
    assert loaded_manifest['network_hash'] == first['network_hash']


def test_seed_manager_named_streams_are_stable():
    first = SeedManager(42)
    second = SeedManager(42)
    assert first.derive('queue', 'placement', 1) == second.derive('queue', 'placement', 1)
    assert first.derive('queue', 'placement', 1) != first.derive('queue', 'placement', 2)


def test_relative_nash_gap_uses_travel_time():
    from queue_sim.find_nash import _relative_gap
    gap, selected = _relative_gap({
        ((0, 1), 'F1'): [
            {'travel_time': 10.0, 'used': True},
            {'travel_time': 9.0, 'used': False},
        ]
    })
    assert gap == pytest.approx(1 / 9)
    assert selected[0] == ((0, 1), 'F1')
