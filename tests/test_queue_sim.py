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


def test_placements_are_order_independent_and_deterministic():
    from src.contracts import (
        canonical_placement,
        enumerate_placements,
        single_swap_neighbors,
    )

    assert canonical_placement([49, 101]) == canonical_placement([101, 49])
    assert enumerate_placements([101, 49, 20], 2) == [
        (49, 101), (20, 101), (20, 49),
    ]
    assert single_swap_neighbors([20, 49], [101, 49, 20]) == [
        (49, 101), (20, 101),
    ]


def test_comparison_tasks_share_seed_and_cover_every_placement(monkeypatch):
    import queue_sim.comparison as comparison
    from src.contracts import enumerate_placements

    seeds = []

    def fake_run(positions, *args, seed=None, **kwargs):
        seeds.append((tuple(positions), seed))
        return float(sum(positions)), {}

    monkeypatch.setattr(comparison, '_run_sim', fake_run)
    comparison._WORKER_INPUTS.update({'data': {}, 'ne': {}})
    candidates = [1, 2, 3, 4]
    placements = [
        placement for size in range(1, 4)
        for placement in enumerate_placements(candidates, size)
    ]
    results = [
        comparison._comparison_task((rep, placement, 16, [], ('n', 'e', 'o'),
                                     '.', 250, 250, 250, 0, 10801, 42))
        for rep in range(2) for placement in placements
    ]
    # Common random numbers: one seed per replication for every placement.
    for rep in range(2):
        rep_seeds = {seed for (placement, seed), result in zip(seeds, results)
                     if result[0] == rep}
        assert len(rep_seeds) == 1
    assert len({seed for _, seed in seeds}) == 2

    paired = comparison._assemble_paired_results(
        results, 2, enumerate_placements(candidates, 3),
    )
    assert len(paired) == 2
    assert all(record['unique_simulations'] == 14 for record in paired)
    assert paired[0]['exhaustive_values'] == [
        float(sum(value)) for value in enumerate_placements(candidates, 3)
    ]


def test_replicated_gap_averages_route_costs_over_replications():
    from queue_sim.comparison import _replicated_gaps

    def details(used_cost, unused_cost):
        return {((0, 1), 'F1'): [
            {'travel_time': used_cost, 'used': True},
            {'travel_time': unused_cost, 'used': False},
        ]}

    gaps = _replicated_gaps([
        (0, (5,), 0.0, 0.0, details(110.0, 100.0)),
        (1, (5,), 0.0, 0.0, details(90.0, 100.0)),
    ])
    # Mean used cost 100 equals mean unused cost 100: no profitable deviation.
    assert gaps['5']['min_normalized'] == pytest.approx(0.0)


def test_queue_search_uses_shared_candidate_order_and_mean_objectives():
    from queue_sim.comparison import _build_queue_search_summary

    values = {
        (101,): 10.0, (49,): 20.0, (20,): 30.0,
        (49, 101): 100.0, (20, 101): 90.0, (20, 49): 80.0,
    }
    paired = [
        {
            'placement_values': {key: value + offset for key, value in values.items()},
            'placement_elapsed_seconds': {key: 1.0 for key in values},
        }
        for offset in (-1.0, 1.0)
    ]
    search = _build_queue_search_summary(
        paired, [101, 49, 20], num_stations=2, single_swap=True,
    )

    assert [item['placement'] for item in search['trace'][:3]] == [
        [101], [49], [20],
    ]
    assert search['greedy']['placement'] == [20, 101]
    assert search['single_swap']['placement'] == [20, 49]
    assert search['exhaustive']['placement'] == [20, 49]


def test_ne_summary_and_cross_model_correlations_are_reportable(tmp_path):
    from queue_sim.comparison import _correlation_summary
    from queue_sim.find_nash import _write_queue_manifest

    assignments = {
        '1': {'converged': True, 'status': 'ok', 'iterations': 3},
        '2': {
            'converged': False, 'status': 'approximate_cycle_state',
            'iterations': 6, 'cycle_length': 2,
            'termination_reason': 'assignment cycle detected at iteration 6',
        },
        '3': {
            'converged': False, 'status': 'nonconverged', 'iterations': 10,
            'failure_reason': 'iteration cap reached',
        },
    }
    path = tmp_path / 'queue_manifest.json'
    _write_queue_manifest(path, {'failed_configurations': {}}, assignments)
    manifest = json.loads(path.read_text())

    assert manifest['status_counts'] == {
        'converged': 1, 'cycle': 1, 'nonconverged': 1, 'failed': 0,
    }
    assert manifest['iteration_statistics']['all']['median'] == 6.0
    assert manifest['cycle_length_statistics']['median'] == 2.0

    correlations = _correlation_summary([1, 2, 3, 4], [10, 20, 40, 30])
    assert correlations['paired_count'] == 4
    assert correlations['pearson'] == pytest.approx(0.8)
    assert correlations['spearman'] == pytest.approx(0.8)


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


def test_queue_network_honors_canonical_link_capacity():
    from queue_sim import QUEUE_SIM_AVAILABLE
    if not QUEUE_SIM_AVAILABLE:
        pytest.skip("queue simulator unavailable")
    from queue_sim.queue_model_EV import Simulation

    nodes = pd.DataFrame([
        {'node_id': 0, 'lon': 0.0, 'lat': 0.0, 'type': 'real', 'node_osmid': 0},
        {'node_id': 1, 'lon': 1.0, 'lat': 0.0, 'type': 'real', 'node_osmid': 1},
    ])
    links = pd.DataFrame([{
        'link_id': 0, 'start_node_id': 0, 'end_node_id': 1,
        'lanes': 1.0, 'length': 100.0, 'maxmph': 25.0, 'fft': 9.0,
        'capacity': 777.0, 'type': 'secondary',
        'geometry': 'LINESTRING (0 0, 1 0)',
    }])
    simulation = Simulation()
    simulation.create_network(nodes, links, pd.DataFrame())

    assert simulation.all_links[0].capacity == pytest.approx(777.0)


def _write_line_network(tmp_path, f2_count=1):
    """Three-node line 0 -> 1 -> 2 with route records for a charger at node 1."""
    nodes = pd.DataFrame([
        {'node_id': i, 'lon': -77.0 + 0.002 * i, 'lat': 38.9, 'node_osmid': i, 'type': 'real'}
        for i in range(3)
    ])
    links = pd.DataFrame([
        {'link_id': i, 'start_node_id': i, 'end_node_id': i + 1, 'type': 'secondary',
         'length': 170.0, 'maxmph': 25.0, 'lanes': 1, 'capacity': 1000.0,
         'start_osmid': i, 'end_osmid': i + 1,
         'geometry': f'LINESTRING ({-77.0 + 0.002 * i} 38.9, {-77.0 + 0.002 * (i + 1)} 38.9)'}
        for i in range(2)
    ])
    od = pd.DataFrame([{
        'origin_node_id': 0, 'destin_node_id': 2, 'origin_osmid': 0, 'destin_osmid': 2,
        'is_EV': True, 'need_to_charge': True, 'current_charge': 0,
        'target_charge': 100, 'go_to_station_id': float('nan'), 'dept_time': 0,
    }] * f2_count)
    paths = [str(tmp_path / name) for name in ('nodes.csv', 'links.csv', 'od.csv')]
    for frame, path in zip((nodes, links, od), paths):
        frame.to_csv(path, index=False)
    flow_data = {(0, 2): {
        'no charging type': [],
        'charging type': [{'route_id': 'c', 'path': [0, 1, 2], 'link_ids': [0, 1],
                           'flow': 1.0, 'station node': 1}],
    }}
    return paths, flow_data


def test_charging_vehicle_enters_station_and_charges(tmp_path):
    from queue_sim import QUEUE_SIM_AVAILABLE
    if not QUEUE_SIM_AVAILABLE:
        pytest.skip("queue simulator unavailable")
    from queue_sim import Runner

    def run(with_station_trip):
        (nodes, links, od), flow_data = _write_line_network(tmp_path)
        runner = Runner(nodes_csv=nodes, links_csv=links, od_csv=od, seed=1)
        runner.create_EV_charging_station_at_node(1, 250, 250, 250, 0)
        if not with_station_trip:
            # Same road route, but the route record names no installed charger.
            flow_data[(0, 2)]['charging type'][0]['station node'] = 99
        runner.init_sq_simulation_with_path_assignment(
            flow_data, {(0, 2): [1]}, {(0, 2): []},
        )
        return runner

    runner = run(True)
    runner.spatial_queue_simulation('t', t_end=2000, output_dir=str(tmp_path / 'out'))
    agent = next(iter(runner.sim.all_agents.values()))
    assert agent.current_charge == agent.target_charge
    # Two 170 m links at 25 mph take ~30 s; charging adds ~100 one-second steps.
    assert agent.arrival_time - agent.dept_time >= 100
    with pytest.raises(ValueError, match='No charging station'):
        run(False)


def test_unused_charging_route_includes_nominal_charging_time(tmp_path):
    from queue_sim import QUEUE_SIM_AVAILABLE
    if not QUEUE_SIM_AVAILABLE:
        pytest.skip("queue simulator unavailable")
    from queue_sim import Runner

    (nodes, links, od), flow_data = _write_line_network(tmp_path)
    flow_data[(0, 2)]['charging type'].append(
        {'route_id': 'unused', 'path': [0, 1, 2], 'link_ids': [0, 1],
         'flow': 0.0, 'station node': 1})
    runner = Runner(nodes_csv=nodes, links_csv=links, od_csv=od, seed=1)
    runner.create_EV_charging_station_at_node(1, 250, 250, 250, 0)
    runner.init_sq_simulation_with_path_assignment(
        flow_data, {(0, 2): [1, 0]}, {(0, 2): []},
    )
    runner.spatial_queue_simulation('t', t_end=2000, output_dir=str(tmp_path / 'out'))
    routes = runner.check_NE(return_details=True)[((0, 2), 'F2')]
    used, unused = routes
    assert runner.nominal_charging_seconds == 100
    # Same physical route: used and unused costs must agree up to station
    # entry/exit seconds, not differ by the ~100 s charging dwell.
    assert abs(used['travel_time'] - unused['travel_time']) < 20


def test_cycle_or_cap_retains_lowest_gap_visited_assignment():
    from queue_sim.find_nash import _retain_lowest_gap_state

    state = {
        'assignments_no': {(0, 1): [3, 0]}, 'assignments_ch': {(0, 1): [0, 3]},
        'best': {
            'gap': 0.03, 'mean_normalized_gap': 0.029, 'route_metrics': {'m': 1},
            'assignments_no': {(0, 1): [2, 1]}, 'assignments_ch': {(0, 1): [1, 2]},
            'iteration': 4,
        },
    }
    _retain_lowest_gap_state(state)
    assert state['assignments_no'] == {(0, 1): [2, 1]}
    assert state['assignments_ch'] == {(0, 1): [1, 2]}
    assert state['final_gap_override'] == 0.03
    assert state['retained_iteration'] == 4


def test_iteration_cap_approximation_is_usable_and_counted_nonconverged(tmp_path):
    from queue_sim.comparison import _assignment_is_usable
    from queue_sim.find_nash import _promote_cycle_result, _write_queue_manifest

    result = {'status': 'approximate_iteration_cap', 'converged': False,
              'iterations': 300, 'termination_reason': 'cap'}
    _promote_cycle_result(result)
    assert result['status'] == 'approximate_iteration_cap'
    assert _assignment_is_usable(result)
    path = tmp_path / 'm.json'
    _write_queue_manifest(path, {'failed_configurations': {}}, {'a': result})
    manifest = json.loads(path.read_text())
    assert manifest['status_counts']['nonconverged'] == 1
    assert manifest['nonconverged_configurations'] == {}
    assert 'a' in manifest['approximate_configurations']
