"""Paper-contract tests for CG-to-route flow recovery."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

from src.traffic_optimizer import Network


def _network(charger_two_connected=True):
    network = Network.__new__(Network)
    network.od_demand = {(0, 3): (2.0, 4.0)}
    network.chargers = np.array([1, 2])
    network.q_c_values = {(0, 2, 0): 1.5, (0, 2, 1): 2.5}
    network.DiGraph = nx.DiGraph()
    edges = [(0, 1, 0), (1, 3, 1)]
    if charger_two_connected:
        edges += [(0, 2, 2), (2, 3, 3)]
    for start, end, link_id in edges:
        network.DiGraph.add_edge(start, end, length=1.0, link_id=link_id)
    network.net = SimpleNamespace(
        nid_to_osmid_dict={value: value for value in range(4)},
        osmid_to_nid_dict={value: value for value in range(4)},
    )
    network.pair_to_link_ids = {
        (start, end): [link_id] for start, end, link_id in edges
    }
    network.pair_to_link_ids.update({(1, 1): [4], (2, 2): [5]})
    flows = {
        0: {'start_node_id': 0, 'end_node_id': 1, 'total_flow': 3.5},
        1: {'start_node_id': 1, 'end_node_id': 3, 'total_flow': 3.5},
        2: {'start_node_id': 0, 'end_node_id': 2, 'total_flow': 2.5},
        3: {'start_node_id': 2, 'end_node_id': 3, 'total_flow': 2.5},
        4: {'start_node_id': 1, 'end_node_id': 1, 'total_flow': 1.5},
        5: {'start_node_id': 2, 'end_node_id': 2, 'total_flow': 2.5},
    }
    return network, flows


def test_route_recovery_preserves_per_charger_cg_demand():
    network, flows = _network()
    recovered = network.reconstruct_route_flows(
        flows, paths_per_od=1, paths_per_oc_cd=1,
        use_od_constraints=True, use_charger_constraints=True,
    )
    charging = recovered[(0, 3)]['charging']
    assert sum(route['flow'] for route in charging[1]) == pytest.approx(1.5)
    assert sum(route['flow'] for route in charging[2]) == pytest.approx(2.5)


def test_route_recovery_rejects_positive_cg_flow_without_a_route():
    network, flows = _network(charger_two_connected=False)
    with pytest.raises(ValueError, match='No feasible recovered route'):
        network.reconstruct_route_flows(
            flows, paths_per_od=1, paths_per_oc_cd=1,
            use_od_constraints=True, use_charger_constraints=True,
        )
