"""Unit tests for merged RoadNet."""
import pytest
import os
import tempfile

import networkx as nx
import pandas as pd

from src.network_artifact import write_network_artifact
from src.road_network import RoadNet


def test_roadnet_init():
    rn = RoadNet('Test Network')
    assert rn.name == 'Test Network'
    assert rn.graph == []
    assert rn.nodes == []
    assert rn.edges == []
    assert rn.exit_node_id == []
    assert rn.demand == []
    assert rn.osmid_to_nid_dict == {}
    assert rn.nid_to_osmid_dict == {}


def test_roadnet_has_all_methods():
    """Merged RoadNet must have methods from both CG and queue versions."""
    rn = RoadNet('Test')
    assert hasattr(rn, 'get_map')
    assert hasattr(rn, 'rearrange_data')
    assert hasattr(rn, 'set_exit')
    assert hasattr(rn, 'create_demand')
    assert hasattr(rn, 'create_demand_with_orig_dest')
    assert hasattr(rn, 'save_data')
    assert hasattr(rn, 'plot_links_and_nodes')


def test_roadnet_save_data_with_dir():
    """save_data should accept optional save_dir."""
    rn = RoadNet('Test Net')
    import pandas as pd
    rn.nodes = pd.DataFrame({'node_id': [0], 'lon': [0.0], 'lat': [0.0], 'type': ['real'], 'node_osmid': [1]})
    rn.edges = pd.DataFrame({'link_id': [0], 'start_node_id': [0], 'end_node_id': [0], 'type': ['real'],
                             'length': [1.0], 'maxmph': [25.0], 'lanes': [1], 'capacity': [1000],
                             'start_osmid': [1], 'end_osmid': [1], 'geometry': ['POINT(0 0)']})
    rn.demand = pd.DataFrame({'origin_node_id': [0], 'destin_node_id': [0],
                              'origin_osmid': [1], 'destin_osmid': ['vn_sink']})
    with tempfile.TemporaryDirectory() as tmpdir:
        rn.save_data(save_dir=tmpdir)
        assert os.path.exists(os.path.join(tmpdir, 'traffic_inputs_test_net_nodes.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'traffic_inputs_test_net_edges.csv'))
        assert os.path.exists(os.path.join(tmpdir, 'traffic_inputs_test_net_od.csv'))


def test_roadnet_loads_canonical_artifact(tmp_path):
    nodes = pd.DataFrame([
        {"node_id": 0, "node_osmid": 10, "lon": -77.0, "lat": 38.9, "type": "real"},
        {"node_id": 1, "node_osmid": 20, "lon": -76.9, "lat": 38.9, "type": "real"},
    ])
    edges = pd.DataFrame([
        {"link_id": 0, "start_node_id": 0, "end_node_id": 1,
         "start_osmid": 10, "end_osmid": 20, "edge_key": 0,
         "type": "primary", "length": 100.0, "maxmph": 25.0,
         "lanes": 1, "capacity": 1000, "travel_time": 9.0,
         "source_edge_ids": '["10|20|0"]',
         "geometry": "LINESTRING (-77 38.9, -76.9 38.9)"},
        {"link_id": 1, "start_node_id": 1, "end_node_id": 0,
         "start_osmid": 20, "end_osmid": 10, "edge_key": 0,
         "type": "primary", "length": 100.0, "maxmph": 25.0,
         "lanes": 1, "capacity": 1000, "travel_time": 9.0,
         "source_edge_ids": '["20|10|0"]',
         "geometry": "LINESTRING (-76.9 38.9, -77 38.9)"},
    ])
    write_network_artifact(nodes, edges, tmp_path / "network")

    road = RoadNet("loaded")
    road.load_artifact(tmp_path / "network")

    assert road.nid_to_osmid_dict == {0: 10, 1: 20}
    assert road.osmid_to_nid_dict == {10: 0, 20: 1}
    assert nx.is_strongly_connected(road.graph)
    assert road.graph[10][20][0]["travel_time"] == 9.0
    assert len(road.nodes) == 2
    assert len(road.edges) == 2
