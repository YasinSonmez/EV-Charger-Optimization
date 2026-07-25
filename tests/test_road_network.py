"""Unit tests for merged RoadNet."""
import pytest
import os
import tempfile

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
