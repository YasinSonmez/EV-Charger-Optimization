"""Multi-OD queue assignment regression test."""

import pandas as pd
import pytest


def test_runner_assigns_all_od_type_groups(tmp_path):
    from queue_sim import QUEUE_SIM_AVAILABLE
    if not QUEUE_SIM_AVAILABLE:
        pytest.skip('queue simulator binary unavailable')
    from queue_sim.runner_EV import Runner

    nodes = pd.DataFrame([
        {'node_id': 0, 'lon': 0.0, 'lat': 0.0, 'node_osmid': 0, 'type': 'real'},
        {'node_id': 1, 'lon': 1.0, 'lat': 0.0, 'node_osmid': 1, 'type': 'real'},
        {'node_id': 2, 'lon': 2.0, 'lat': 0.0, 'node_osmid': 2, 'type': 'real'},
    ])
    edges = pd.DataFrame([
        {'link_id': 0, 'start_node_id': 0, 'end_node_id': 1, 'type': 'primary', 'length': 1.0, 'maxmph': 25.0, 'lanes': 1, 'capacity': 1900, 'geometry': 'LINESTRING (0 0, 1 0)'},
        {'link_id': 1, 'start_node_id': 1, 'end_node_id': 2, 'type': 'primary', 'length': 1.0, 'maxmph': 25.0, 'lanes': 1, 'capacity': 1900, 'geometry': 'LINESTRING (1 0, 2 0)'},
        {'link_id': 2, 'start_node_id': 2, 'end_node_id': 0, 'type': 'primary', 'length': 1.0, 'maxmph': 25.0, 'lanes': 1, 'capacity': 1900, 'geometry': 'LINESTRING (2 0, 0 0)'},
    ])
    od = pd.DataFrame([
        {'origin_node_id': 0, 'destin_node_id': 2, 'is_EV': False, 'need_to_charge': False, 'current_charge': 0, 'target_charge': 100, 'go_to_station_id': None},
        {'origin_node_id': 0, 'destin_node_id': 2, 'is_EV': True, 'need_to_charge': True, 'current_charge': 0, 'target_charge': 100, 'go_to_station_id': 1},
        {'origin_node_id': 2, 'destin_node_id': 0, 'is_EV': False, 'need_to_charge': False, 'current_charge': 0, 'target_charge': 100, 'go_to_station_id': None},
        {'origin_node_id': 2, 'destin_node_id': 0, 'is_EV': True, 'need_to_charge': True, 'current_charge': 0, 'target_charge': 100, 'go_to_station_id': 1},
    ])
    nodes_path = tmp_path / 'nodes.csv'
    edges_path = tmp_path / 'edges.csv'
    od_path = tmp_path / 'od.csv'
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    od.to_csv(od_path, index=False)

    runner = Runner(str(edges_path), str(nodes_path), str(od_path), seed=42)
    data = {
        (0, 2): {
            'no charging type': [{'path': [0, 1, 2], 'flow': 1}],
            'charging type': [{'path': [0, 1, 2], 'flow': 1, 'station node': 1}],
        },
        (2, 0): {
            'no charging type': [{'path': [2, 0], 'flow': 1}],
            'charging type': [{'path': [2, 0], 'flow': 1, 'station node': 1}],
        },
    }
    runner.init_sq_simulation_with_path_assignment(
        data,
        {(0, 2): [1], (2, 0): [1]},
        {(0, 2): [1], (2, 0): [1]},
    )
    assert len(runner.route_groups) == 4
    assert sum(len(group['agent_ids']) for group in runner.route_groups) == 4
    assert {group['od_pair'] for group in runner.route_groups} == {(0, 2), (2, 0)}
