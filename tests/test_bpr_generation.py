"""BPR worker regression tests for canonical-network edge cases."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from queue_sim import QUEUE_SIM_AVAILABLE
from queue_sim.bpr_data_generator import (
    BPR_CALIBRATION_VERSION,
    _bpr_link_worker,
    _capacity_fraction_flow_plan,
    _configure_probe_capacity,
    _entry_wait_inclusive_target_times,
    _free_flow_time,
    _proxy_row,
    _synthetic_boundary_overlay,
)
from src.traffic_optimizer import Network
from src.network_artifact import write_network_artifact


def test_capacity_fraction_plan_uses_per_link_capacity():
    link = {'link_id': 4, 'lanes': 2, 'capacity': 2000}
    capacity, plan = _capacity_fraction_flow_plan(
        link,
        [0.0, 0.5, 1.0, 2.0],
        capacity_source='simulator',
        capacity_per_lane=1900,
        calibration_window_hours=1.0,
    )
    assert capacity == 3800.0
    assert [item['flow_rate'] for item in plan] == [0.0, 1900.0, 3800.0, 7600.0]
    assert [item['demand_count'] for item in plan] == [0, 1900, 3800, 7600]


def test_capacity_fraction_plan_allows_positive_only_sweep():
    link = {'link_id': 4, 'lanes': 1, 'capacity': 1000}
    capacity, plan = _capacity_fraction_flow_plan(
        link,
        [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0],
        capacity_source='simulator',
        capacity_per_lane=1900,
        calibration_window_hours=0.1,
    )
    assert capacity == 1900.0
    assert plan[0]['fraction'] == 0.1
    assert plan[-1]['fraction'] == 10.0
    assert all(item['demand_count'] > 0 for item in plan)


def test_free_flow_time_uses_mph_to_mps_conversion():
    # 100 m at 25 mph is approximately 8.95 seconds.
    assert _free_flow_time({'link_id': 0, 'length': 100.0, 'maxmph': 25.0}) == pytest.approx(8.95, rel=1e-3)


def test_proxy_row_uses_the_same_free_flow_conversion():
    row = _proxy_row(
        {'link_id': 0, 'length': 100.0, 'maxmph': 25.0},
        [1.0, 2.0],
    )
    assert row['observation_source'] == 'proxy'
    assert row['y_vector'] == pytest.approx([_free_flow_time({'length': 100.0, 'maxmph': 25.0})] * 2)


@pytest.mark.parametrize(
    ('missing_predecessor', 'missing_successor'),
    [(True, False), (False, True), (True, True)],
)
def test_synthetic_boundary_overlay_is_worker_local_and_deterministic(
    tmp_path, missing_predecessor, missing_successor
):
    nodes = pd.DataFrame(
        {
            'node_id': [0, 1, 2],
            'node_osmid': [0, 1, 2],
            'lon': [0.0, 1.0, 2.0],
            'lat': [0.0, 0.0, 0.0],
            'type': ['real', 'real', 'real'],
        }
    )
    edges = pd.DataFrame(
        {
            'link_id': [0, 1],
            'start_node_id': [0, 1],
            'end_node_id': [1, 2],
            'type': ['road', 'road'],
            'length': [100.0, 100.0],
            'maxmph': [25.0, 25.0],
            'lanes': [1, 1],
            'capacity': [1900.0, 1900.0],
            'start_osmid': [0, 1],
            'end_osmid': [1, 2],
            'edge_key': ['a', 'b'],
            'geometry': ['LINESTRING (0 0, 1 0)', 'LINESTRING (1 0, 2 0)'],
        }
    )
    artifact_dir = tmp_path / 'network'
    write_network_artifact(nodes, edges, artifact_dir)
    target = edges.iloc[0].to_dict()
    predecessor = None if missing_predecessor else edges.iloc[0]
    successor = None if missing_successor else edges.iloc[1]
    flow_plan = [{'demand_count': 1}, {'demand_count': 250}]

    first = _synthetic_boundary_overlay(
        str(artifact_dir / 'nodes.csv'),
        str(artifact_dir / 'edges.csv'),
        target,
        predecessor,
        successor,
        flow_plan,
        str(tmp_path / 'link_0'),
    )
    second = _synthetic_boundary_overlay(
        str(artifact_dir / 'nodes.csv'),
        str(artifact_dir / 'edges.csv'),
        target,
        predecessor,
        successor,
        flow_plan,
        str(tmp_path / 'link_1'),
    )
    assert first[4] == second[4]
    assert first[4]['context_mode'] == 'synthetic_boundary'
    assert len(first[4]['synthetic_link_ids']) == int(missing_predecessor) + int(missing_successor)
    overlay_edges = pd.read_csv(first[1])
    assert set(overlay_edges['link_id']) >= {0, 1}
    assert all(int(value) > int(edges['link_id'].max()) for value in first[4]['synthetic_link_ids'])
    synthetic = overlay_edges[overlay_edges['type'] == 'synthetic_boundary']
    assert (synthetic['capacity'] >= 10 * 3600 * 250).all()
    # Canonical artifact files remain unchanged and contain no synthetic IDs.
    canonical_edges = pd.read_csv(artifact_dir / 'edges.csv')
    assert set(canonical_edges['link_id']) == {0, 1}
    assert 'synthetic_boundary' not in set(canonical_edges['type'])


def test_capacity_fraction_plan_rejects_duplicate_agent_counts():
    with pytest.raises(ValueError, match='duplicate simulator demand counts'):
        _capacity_fraction_flow_plan(
            {'link_id': 4, 'lanes': 1},
            [0.0, 0.1, 0.2],
            capacity_source='simulator',
            capacity_per_lane=1.0,
            calibration_window_hours=1.0,
        )


def test_probe_continuation_is_strictly_nonbinding():
    runner = SimpleNamespace(links_df=pd.DataFrame({
        'link_id': [1, 2], 'lanes': [1.0, 3.0],
    }))
    _configure_probe_capacity(
        runner, {'lanes': 2.0}, [{'link_id': 1}, {'link_id': 2}],
        capacity_multiplier=10.0,
    )
    assert runner.links_df['lanes'].tolist() == [20.0, 20.0]


def test_entry_wait_inclusive_cost_subtracts_only_continuation_time():
    runner = SimpleNamespace(sim=SimpleNamespace(
        all_links={
            7: SimpleNamespace(completed_travel_time_list=[[0, 10.0], [1, 12.0]]),
            8: SimpleNamespace(completed_travel_time_list=[[0, 3.0], [1, 4.0]]),
        },
        all_agents={
            0: SimpleNamespace(arrival_time=15.0, dept_time=0.0),
            1: SimpleNamespace(arrival_time=20.0, dept_time=2.0),
        },
    ))
    values = _entry_wait_inclusive_target_times(runner, 7, [8])
    assert values.tolist() == [12.0, 14.0]


@pytest.mark.skipif(not QUEUE_SIM_AVAILABLE, reason="queue simulator unavailable")
def test_strict_worker_uses_cohort_units_and_simulates_zero_probe(tmp_path):
    nodes = pd.DataFrame({
        'node_id': [0, 1, 2], 'node_osmid': [0, 1, 2],
        'lon': [0.0, 1.0, 2.0], 'lat': [0.0, 0.0, 0.0],
        'type': ['real', 'real', 'real'],
    })
    edges = pd.DataFrame({
        'link_id': [0, 1], 'start_node_id': [0, 1],
        'end_node_id': [1, 2], 'type': ['road', 'road'],
        'length': [100.0, 100.0], 'maxmph': [25.0, 25.0],
        'lanes': [1, 1], 'capacity': [1900.0, 1900.0],
        'start_osmid': [0, 1], 'end_osmid': [1, 2],
        'edge_key': ['target', 'continuation'],
        'geometry': ['LINESTRING (0 0, 1 0)', 'LINESTRING (1 0, 2 0)'],
    })
    artifact = tmp_path / 'network'
    write_network_artifact(nodes, edges, artifact)
    flow_spec = {
        'mode': 'capacity_fraction_strict',
        'flow_fractions': [0.0, 0.5, 2.0],
        'capacity_source': 'simulator', 'capacity_per_lane': 1900.0,
        'calibration_window_hours': 0.01, 'route_mode': 'link_probe',
        'simulation_horizon': 1000, 'resume': False,
        'calibration_version': BPR_CALIBRATION_VERSION,
        'probe_continuation_capacity_multiplier': 10.0,
    }
    result = _bpr_link_worker((
        edges.iloc[0].to_dict(), str(artifact / 'nodes.csv'),
        str(artifact / 'edges.csv'), flow_spec, str(tmp_path / 'sweeps'), 42,
    ))
    assert result['errors'] == []
    assert result['calibration_version'] == BPR_CALIBRATION_VERSION
    assert result['capacity_rate'] == 1900.0
    assert result['capacity_count'] == 19.0
    assert result['x_vector'] == [0.0, 10.0, 38.0]
    assert result['observations'][0]['simulated_demand_count'] == 1
    assert result['observations'][0]['status'] == 'simulated_zero_flow_probe'
    assert result['observations'][1]['requested_flow_vph'] == 950.0
    assert all(
        item['measurement'] == 'offered_demand_entry_wait_inclusive'
        for item in result['observations']
    )
    assert result['y_vector'][-1] > result['y_vector'][0]


def test_cg_provenance_policy_reports_or_rejects_active_degraded_links():
    network = Network.__new__(Network)
    network.cg_fit_policy = 'allow_degraded'
    network.active_link_indices = [0, 1]
    network.parameter_fit_results = pd.DataFrame({
        'link_id': [0, 1],
        'fit_status': ['full', 'proxy'],
        'observation_source': ['simulated_contextual', 'proxy'],
        'a_fit': [1.0, 0.0],
        'b_fit': [1.0, 0.0],
        'cap_fit': [100.0, 1.0],
        'fft_fit': [10.0, 10.0],
    })
    network._validate_active_bpr_provenance()
    assert network.bpr_provenance['degraded'] is True
    assert network.bpr_provenance['degraded_link_ids'] == [1]

    network.cg_fit_policy = 'reject_proxy_or_constant'
    with pytest.raises(ValueError, match='active degraded BPR links'):
        network._validate_active_bpr_provenance()


def test_cg_provenance_excludes_derived_charger_self_links_from_road_policy():
    network = Network.__new__(Network)
    network.cg_fit_policy = 'validated_only'
    network.active_link_indices = [0, 1]
    network.parameter_fit_results = pd.DataFrame({
        'link_id': [0, 1],
        'fit_status': ['full', 'derived_charger_link'],
        'observation_source': ['simulated_contextual', 'derived_charger_self_link'],
        'a_fit': [1.0, 0.0],
        'b_fit': [1.0, 1.0],
        'cap_fit': [100.0, 0.527778],
        'fft_fit': [10.0, 100.0],
    })
    network._validate_active_bpr_provenance()
    assert network.bpr_provenance['degraded'] is False
    assert network.bpr_provenance['observation_source_counts']['derived_charger_self_link'] == 1


@pytest.mark.skipif(not QUEUE_SIM_AVAILABLE, reason="queue simulator unavailable")
def test_bpr_worker_measures_link_without_straight_ahead_continuation(tmp_path):
    """A two-link directed cycle must not fail merely because continuation is a U-turn."""
    nodes = pd.DataFrame(
        {
            "node_id": [0, 1],
            "node_osmid": [0, 1],
            "lon": [0.0, 1.0],
            "lat": [0.0, 0.0],
            "type": ["real", "real"],
        }
    )
    edges = pd.DataFrame(
        {
            "link_id": [0, 1],
            "start_node_id": [0, 1],
            "end_node_id": [1, 0],
            "type": ["road", "road"],
            "length": [100.0, 100.0],
            "maxmph": [25.0, 25.0],
            "lanes": [1, 1],
            "capacity": [1900, 1900],
            "start_osmid": [0, 1],
            "end_osmid": [1, 0],
            "edge_key": ["forward", "reverse"],
            "geometry": ["LINESTRING (0 0, 1 0)", "LINESTRING (1 0, 0 0)"],
        }
    )
    artifact_dir = tmp_path / "network"
    write_network_artifact(nodes, edges, artifact_dir)

    result = _bpr_link_worker(
        (
            edges.iloc[0].to_dict(),
            str(artifact_dir / "nodes.csv"),
            str(artifact_dir / "edges.csv"),
            [1],
            str(tmp_path / "sweeps"),
            11,
        )
    )

    assert result["errors"] == []
    assert result["x_vector"] == [1.0]
    assert result["y_vector"] == [9.0]
    assert result["observations"][0]["replications"] == 1
    assert result["complete"] is True


def test_historical_worker_uses_synthetic_boundary_for_single_boundary_link(tmp_path, monkeypatch):
    nodes = pd.DataFrame(
        {
            'node_id': [0, 1],
            'node_osmid': [0, 1],
            'lon': [0.0, 1.0],
            'lat': [0.0, 0.0],
            'type': ['real', 'real'],
        }
    )
    edges = pd.DataFrame(
        {
            'link_id': [0],
            'start_node_id': [0],
            'end_node_id': [1],
            'type': ['road'],
            'length': [100.0],
            'maxmph': [25.0],
            'lanes': [1],
            'capacity': [1900],
            'start_osmid': [0],
            'end_osmid': [1],
            'edge_key': ['forward'],
            'geometry': ['LINESTRING (0 0, 1 0)'],
        }
    )
    artifact_dir = tmp_path / 'network'
    write_network_artifact(nodes, edges, artifact_dir)

    class FakeRunner:
        def __init__(self, nodes_csv, links_csv, od_csv, seed=None):
            self.nodes_df = pd.read_csv(nodes_csv)
            self.links_df = pd.read_csv(links_csv)
            self.od_df = pd.read_csv(od_csv)
            self.sim = SimpleNamespace(all_links={})

        def find_sa_in_link(self, link):
            rows = self.links_df.loc[
                self.links_df['end_node_id'].astype(int) == int(link['start_node_id'])
            ]
            return rows.iloc[0] if not rows.empty else None

        def find_sa_out_link(self, link):
            rows = self.links_df.loc[
                self.links_df['start_node_id'].astype(int) == int(link['end_node_id'])
            ]
            return rows.iloc[0] if not rows.empty else None

        def add_charging_info(self, *_args):
            return None

        def init_sq_simulation_for_bpr_function_fitting_V2(self, link, *_args):
            demand = len(self.od_df)
            self.sim.all_links[0] = SimpleNamespace(
                completed_travel_time_list=[9.0] * demand,
                tot_entering_vehs=demand,
                ave_travel_time=9.0,
                queue_veh=[],
                run_veh=[],
            )

        def spatial_queue_simulation(self, *_args, **_kwargs):
            return None

        def return_traffic_data(self, *_args, **_kwargs):
            return pd.DataFrame({'link_id': [0], 'flow': [1.0], 'travel_time': [9.0]})

    monkeypatch.setattr('queue_sim.bpr_data_generator.Runner', FakeRunner)
    result = _bpr_link_worker(
        (
            edges.iloc[0].to_dict(),
            str(artifact_dir / 'nodes.csv'),
            str(artifact_dir / 'edges.csv'),
            {
                'mode': 'historical_artifact_compatible',
                'flow_levels': [1],
                'route_mode': 'contextual',
                'missing_context_policy': 'synthetic_boundary',
                'synthetic_context_capacity_multiplier': 10.0,
                'synthetic_context_length_m': 1.0,
            },
            str(tmp_path / 'sweeps'),
            17,
        )
    )
    assert result['errors'] == []
    assert result['observation_source'] == 'simulated_synthetic_context'
    assert result['context_mode'] == 'synthetic_boundary'
    assert len(result['synthetic_link_ids']) == 2
    assert len(result['x_vector']) == len(result['y_vector']) == 1
    overlay_edges = pd.read_csv(tmp_path / 'sweeps' / 'link_0' / 'edges_with_synthetic_context.csv')
    assert len(overlay_edges) == 3
    assert int(overlay_edges.loc[overlay_edges['link_id'] == 0, 'link_id'].iloc[0]) == 0
