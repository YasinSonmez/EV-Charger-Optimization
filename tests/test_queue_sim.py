"""Unit tests for queue simulation availability and basic imports."""
import json
import pickle
from types import SimpleNamespace

import pandas as pd
import pytest


def test_queue_sim_import():
    """Test that queue_sim can be imported (may be unavailable on non-macOS)."""
    from queue_sim import QUEUE_SIM_AVAILABLE, _QUEUE_SIM_ERROR
    if QUEUE_SIM_AVAILABLE:
        from queue_sim import Runner
        assert Runner is not None
    else:
        pytest.skip(f"Queue sim not available on this platform: {_QUEUE_SIM_ERROR}")


def test_queue_sim_helpers():
    """Test the helper functions from find_nash and comparison modules."""
    from queue_sim.find_nash import _collapse_repeats, _rounded_counts, _prune_flow_data
    assert _collapse_repeats([1, 1, 2, 2, 3, 3, 1]) == [1, 2, 3, 1]
    assert _collapse_repeats([]) == []
    assert _collapse_repeats([5]) == [5]

    routes = [{'flow': 10.0}, {'flow': 20.0}, {'flow': 30.0}]
    counts = _rounded_counts(routes, 6)
    assert sum(counts) == 6
    assert len(counts) == 3

    from queue_sim.comparison import _placement_seed
    s1 = _placement_seed(0, [14, 20])
    s2 = _placement_seed(0, [20, 14])
    s3 = _placement_seed(1, [14, 20])
    assert s1 == s2
    assert s1 != s3


def test_cycle_assignment_is_explicitly_approximate_and_comparison_usable():
    from queue_sim.comparison import _assignment_is_usable
    from queue_sim.find_nash import _promote_cycle_result

    result = {
        'status': 'nonconverged',
        'converged': False,
        'failure_reason': 'assignment cycle detected at iteration 4',
        'assignments': {'F1': {(0, 1): [1]}, 'F2': {(0, 1): [1]}},
    }
    _promote_cycle_result(result, gap_verified=False)

    assert result['status'] == 'approximate_cycle_state'
    assert result['approximate_equilibrium'] is True
    assert result['converged'] is False
    assert result['exact_ne_eligible'] is False
    assert result['retained_state_gap_verified'] is False
    assert result['failure_reason'] is None
    assert _assignment_is_usable(result)


def test_unlabeled_nonconverged_assignment_remains_unusable():
    from queue_sim.comparison import _assignment_is_usable

    assert not _assignment_is_usable({
        'status': 'nonconverged',
        'converged': False,
    })


def test_resume_promotes_legacy_cycle_artifact_without_resimulation(tmp_path):
    from queue_sim.find_nash import _reuse_saved_cycle_assignments

    work_dir = tmp_path / 'queue'
    work_dir.mkdir()
    legacy = {
        '75': {
            'status': 'nonconverged',
            'converged': False,
            'failure_reason': 'assignment cycle detected at iteration 4',
            'final_gap': 0.2,
            'network_hash': 'network-1',
            'assignments': {'F1': {(0, 1): [1]}, 'F2': {(0, 1): [1]}},
        }
    }
    with (work_dir / 'NE_path_assignments.pkl').open('wb') as handle:
        pickle.dump(legacy, handle)
    (work_dir / 'queue_manifest.json').write_text(json.dumps({
        'network_hash': 'network-1',
        'configuration_count': 1,
        'failed_configurations': {},
        'nonconverged_configurations': {
            '75': 'assignment cycle detected at iteration 4'
        },
    }))

    reused = _reuse_saved_cycle_assignments(work_dir, 'network-1', 1)

    assert reused is not None
    _, assignments = reused
    assert assignments['75']['status'] == 'approximate_cycle_state'
    assert assignments['75']['retained_state_gap_verified'] is False
    manifest = json.loads((work_dir / 'queue_manifest.json').read_text())
    assert manifest['nonconverged_configurations'] == {}
    assert manifest['approximate_configurations'] == {
        '75': 'assignment cycle detected at iteration 4'
    }
    assert manifest['exact_ne_eligible'] is False


def test_unused_route_uses_current_link_cost_not_free_flow():
    from queue_sim import QUEUE_SIM_AVAILABLE
    if not QUEUE_SIM_AVAILABLE:
        pytest.skip("queue simulator unavailable")
    from queue_sim.runner_EV import Runner

    runner = Runner.__new__(Runner)
    runner.sim = SimpleNamespace(
        all_agents={},
        all_links={
            10: SimpleNamespace(ave_travel_time=18.0, fft=5.0),
            11: SimpleNamespace(ave_travel_time=12.0, fft=4.0),
        },
        resolve_link_id=lambda start, end: {(0, 1): 10, (1, 2): 11}[(start, end)],
    )
    runner.route_groups = [{
        'od_pair': (0, 2), 'vehicle_type': 'F1',
        'paths': [[(0, 1), (1, 2)]],
        'route_agent_ids': {0: []},
        'entries': [{'station_cost': 0.0}],
        'route_ids': ['unused'],
    }]
    details = runner._check_route_details()
    assert details[((0, 2), 'F1')][0]['travel_time'] == pytest.approx(30.0)


def test_station_at_node_accepts_single_incoming_link():
    """A valid one-incoming-road candidate must support a virtual station."""
    from queue_sim import QUEUE_SIM_AVAILABLE
    if not QUEUE_SIM_AVAILABLE:
        pytest.skip("queue simulator unavailable")
    from queue_sim.runner_EV import Runner

    runner = Runner.__new__(Runner)
    runner.nodes_df = pd.DataFrame([
        {'node_id': 0, 'lon': 0.0, 'lat': 0.0, 'node_osmid': 0, 'type': 'real'},
        {'node_id': 1, 'lon': 1.0, 'lat': 0.0, 'node_osmid': 1, 'type': 'real'},
    ])
    runner.links_df = pd.DataFrame([{
        'link_id': 0, 'start_node_id': 0, 'end_node_id': 1,
        'type': 'secondary', 'length': 100.0, 'maxmph': 25.0,
        'lanes': 1, 'capacity': 1900.0,
        'start_osmid': 0, 'end_osmid': 1,
        'geometry': 'LINESTRING (0 0, 1 0)',
    }])
    runner.charging_stations_df = pd.DataFrame()

    runner.create_EV_charging_station_at_node(
        station_node_id=1,
        ent_capacity=250,
        charging_capacity=250,
        exit_capacity=250,
        cost=0,
    )

    assert len(runner.charging_stations_df) == 1
    assert set(runner.links_df['type']) == {
        'secondary', 'In_Station', 'Out_Station',
    }
    station_node = runner.nodes_df.loc[runner.nodes_df['type'] == 'Station'].iloc[0]
    assert station_node['lon'] == pytest.approx(1.0)
    assert station_node['lat'] != pytest.approx(0.0)
