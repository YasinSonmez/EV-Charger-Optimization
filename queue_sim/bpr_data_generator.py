"""Parallel queue-based BPR data generation for the canonical network."""

from __future__ import annotations

import json
import os
import time
import traceback
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from queue_sim import Runner, QUEUE_SIM_AVAILABLE
from src.contracts import SeedManager
from src.network_artifact import load_network_artifact
from src.road_network import RoadNet
from src.run_state import atomic_write_json, available_cpus


DEFAULT_FLOW_FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
DEFAULT_SIMULATOR_CAPACITY_PER_LANE = 1900.0  # vehicles/hour/lane


def _effective_capacity_rate(link_record, capacity_source='simulator', capacity_per_lane=DEFAULT_SIMULATOR_CAPACITY_PER_LANE):
    """Return the capacity used to construct capacity-fraction flow levels."""
    if capacity_source == 'artifact':
        capacity = float(link_record.get('capacity', 0.0))
    elif capacity_source == 'simulator':
        lanes = float(link_record.get('lanes', 1.0) or 1.0)
        capacity = lanes * float(capacity_per_lane)
    else:
        raise ValueError("capacity_source must be 'simulator' or 'artifact'")
    if not np.isfinite(capacity) or capacity <= 0:
        raise ValueError(f"link {link_record.get('link_id')} has invalid capacity {capacity}")
    return capacity


def _capacity_fraction_flow_plan(link_record, flow_fractions,
                                 capacity_source='simulator',
                                 capacity_per_lane=DEFAULT_SIMULATOR_CAPACITY_PER_LANE,
                                 calibration_window_hours=0.1):
    """Build (BPR flow rate, simulator agent count) pairs for one link.

    BPR x-values are hourly-equivalent flows.  The queue backend consumes an
    integer number of agents, so the configured calibration window converts
    each rate to a demand count.  The default one-hour window preserves the
    paper's flow units while keeping the conversion explicit.
    """
    if calibration_window_hours <= 0:
        raise ValueError('calibration_window_hours must be positive')
    fractions = [float(value) for value in flow_fractions]
    if not fractions or any(not np.isfinite(value) or value < 0 for value in fractions):
        raise ValueError('flow_fractions must contain finite non-negative values')
    if len(set(fractions)) != len(fractions):
        raise ValueError('flow_fractions must be unique')
    capacity_rate = _effective_capacity_rate(
        link_record, capacity_source=capacity_source,
        capacity_per_lane=capacity_per_lane,
    )
    plan = []
    for fraction in fractions:
        flow_rate = fraction * capacity_rate
        demand_count = int(round(flow_rate * float(calibration_window_hours)))
        if fraction > 0 and demand_count < 1:
            demand_count = 1
        plan.append({
            'fraction': fraction,
            'flow_rate': float(flow_rate),
            'demand_count': demand_count,
        })
    if len({item['demand_count'] for item in plan}) != len(plan):
        raise ValueError(
            f"capacity-fraction levels collapse to duplicate simulator demand counts "
            f"for link {link_record.get('link_id')}; increase the calibration window"
        )
    return capacity_rate, plan


def _free_flow_time(link_record):
    # ``maxmph`` is stored in miles/hour.  The queue simulator converts it
    # to metres/second by dividing by 2.2369; using multiplication here
    # creates a free-flow anchor roughly five times too small.
    speed_mps = float(link_record.get('maxmph', 0.0)) / 2.2369
    length = float(link_record.get('length', 0.0))
    if speed_mps <= 0 or length < 0:
        raise ValueError(f"link {link_record.get('link_id')} has invalid free-flow metadata")
    return length / speed_mps


def _departure_schedule(demand_count, calibration_window_hours):
    """Return an integer-second departure schedule over the calibration window.

    The queue simulator advances in one-second steps, so multiple vehicles may
    share a second at high flow.  This preserves the requested rate without
    requiring sub-second simulator changes.
    """
    demand_count = int(demand_count)
    if demand_count <= 0:
        return []
    duration_seconds = max(1, int(round(float(calibration_window_hours) * 3600.0)))
    return np.floor(
        np.arange(demand_count, dtype=float) * duration_seconds / demand_count
    ).astype(int).tolist()


def _write_single_od(path, origin, destination, demand, departure_times=None):
    if departure_times is None:
        departure_times = [0] * int(demand)
    if len(departure_times) != int(demand):
        raise ValueError('departure_times must contain exactly demand entries')
    pd.DataFrame([{
        'origin_node_id': int(origin),
        'destin_node_id': int(destination),
        'origin_osmid': int(origin),
        'destin_osmid': int(destination),
        'dept_time': int(departure_time),
    } for departure_time in departure_times]).to_csv(path, index=False)


def _least_turn_successor(runner, link_record):
    """Return the least-turn outgoing edge, including a U-turn if necessary.

    BPR calibration needs a downstream receiving link to exercise the target
    link's capacity.  This is a calibration-only continuation; it is not used
    for route choice or reported as part of the target link's travel time.
    """
    end_node = runner.nodes_df.loc[
        runner.nodes_df.node_id == int(link_record['end_node_id'])
    ]
    start_node = runner.nodes_df.loc[
        runner.nodes_df.node_id == int(link_record['start_node_id'])
    ]
    if end_node.empty or start_node.empty:
        raise ValueError(f"link {link_record['link_id']} has missing endpoint metadata")
    in_vec = (
        float(start_node.iloc[0].lat) - float(end_node.iloc[0].lat),
        float(start_node.iloc[0].lon) - float(end_node.iloc[0].lon),
    )
    candidates = runner.links_df.loc[
        runner.links_df.start_node_id == int(link_record['end_node_id'])
    ].sort_values('link_id', kind='mergesort')
    if candidates.empty:
        return None
    scored = []
    for _, candidate in candidates.iterrows():
        next_node = runner.nodes_df.loc[
            runner.nodes_df.node_id == int(candidate.end_node_id)
        ]
        if next_node.empty:
            continue
        out_vec = (
            float(end_node.iloc[0].lat) - float(next_node.iloc[0].lat),
            float(end_node.iloc[0].lon) - float(next_node.iloc[0].lon),
        )
        angle = float(np.degrees(np.arctan2(
            in_vec[0] * out_vec[1] - in_vec[1] * out_vec[0],
            in_vec[0] * out_vec[0] + in_vec[1] * out_vec[1],
        )))
        scored.append((abs(angle), int(candidate.link_id), candidate))
    if not scored:
        return None
    return min(scored, key=lambda item: (item[0], item[1]))[2]


def _probe_continuation(runner, link_record):
    """Build a valid downstream probe continuation for one target link.

    A U-turn can return to the target's origin.  Because the queue backend
    treats reaching the configured destination as immediate arrival, append a
    second outgoing edge in that case so the target link is not skipped.
    """
    first = _least_turn_successor(runner, link_record)
    if first is None:
        return []
    continuation = [first]
    target_origin = int(link_record['start_node_id'])
    if int(first['end_node_id']) != target_origin:
        return continuation

    candidates = runner.links_df.loc[
        (runner.links_df.start_node_id == int(first['end_node_id']))
        & (runner.links_df.link_id != int(link_record['link_id']))
    ].sort_values('link_id', kind='mergesort')
    if candidates.empty:
        raise RuntimeError(
            f"link {link_record['link_id']} has a U-turn-only continuation "
            "that returns to its origin and no downstream probe edge"
        )
    continuation.append(candidates.iloc[0])
    if int(continuation[-1]['end_node_id']) == target_origin:
        raise RuntimeError(
            f"link {link_record['link_id']} probe continuation returns to its origin"
        )
    return continuation


def _configure_probe_capacity(runner, target_record, continuation_links):
    """Prevent the calibration-only continuation from becoming a bottleneck."""
    target_lanes = float(target_record.get('lanes', 1.0) or 1.0)
    continuation_ids = {
        int(item['link_id']) for item in continuation_links
    }
    if not continuation_ids:
        return
    mask = runner.links_df.link_id.astype(int).isin(continuation_ids)
    runner.links_df.loc[mask, 'lanes'] = runner.links_df.loc[mask, 'lanes'].astype(float).clip(
        lower=target_lanes
    )


def _synthetic_boundary_overlay(nodes_csv, edges_csv, link_record,
                                predecessor, successor, flow_plan,
                                output_dir, capacity_multiplier=10.0,
                                synthetic_length_m=1.0):
    """Create a worker-local source/sink overlay for a boundary target link.

    The overlay is deliberately derived from, but never written over, the
    canonical artifact.  It exists only to give the queue simulator a valid
    entry/exit context while the canonical target link remains the measured
    link.  Synthetic links are given enough hourly capacity to move the full
    all-at-once calibration demand without becoming the bottleneck.
    """
    if predecessor is not None and successor is not None:
        return nodes_csv, edges_csv, predecessor, successor, {
            'context_mode': 'contextual',
            'synthetic_link_ids': [],
            'missing_context': [],
        }

    nodes = pd.read_csv(nodes_csv)
    edges = pd.read_csv(edges_csv)
    target_id = int(link_record['link_id'])
    missing = []
    if predecessor is None:
        missing.append('predecessor')
    if successor is None:
        missing.append('successor')

    node_ids = pd.to_numeric(nodes['node_id'], errors='coerce').astype(int)
    edge_ids = pd.to_numeric(edges['link_id'], errors='coerce').astype(int)
    max_node_id = int(node_ids.max())
    max_edge_id = int(edge_ids.max())
    # The IDs are deterministic for a target link and live strictly above the
    # canonical ranges.  The overlay itself is still isolated per worker.
    source_node_id = max_node_id + 1 + 2 * target_id
    sink_node_id = max_node_id + 2 + 2 * target_id
    source_link_id = max_edge_id + 1 + 2 * target_id
    sink_link_id = max_edge_id + 2 + 2 * target_id

    start_node = nodes.loc[node_ids == int(link_record['start_node_id'])]
    end_node = nodes.loc[node_ids == int(link_record['end_node_id'])]
    if start_node.empty or end_node.empty:
        raise ValueError(f'target link {target_id} has missing endpoint nodes')
    start_lon = float(start_node.iloc[0]['lon'])
    start_lat = float(start_node.iloc[0]['lat'])
    end_lon = float(end_node.iloc[0]['lon'])
    end_lat = float(end_node.iloc[0]['lat'])
    dx = end_lon - start_lon
    dy = end_lat - start_lat
    norm = max(float(np.hypot(dx, dy)), np.finfo(float).eps)
    step = 1e-5
    source_lon = start_lon - step * dx / norm
    source_lat = start_lat - step * dy / norm
    sink_lon = end_lon + step * dx / norm
    sink_lat = end_lat + step * dy / norm

    target_lanes = float(link_record.get('lanes', 1.0) or 1.0)
    if not np.isfinite(target_lanes) or target_lanes <= 0:
        target_lanes = 1.0
    target_capacity = float(link_record.get('capacity', np.nan))
    if not np.isfinite(target_capacity) or target_capacity <= 0:
        target_capacity = target_lanes * DEFAULT_SIMULATOR_CAPACITY_PER_LANE
    maximum_demand = max(
        [int(item.get('demand_count', 0)) for item in flow_plan] or [0]
    )
    desired_capacity = max(
        float(capacity_multiplier) * target_capacity,
        float(capacity_multiplier) * 3600.0 * maximum_demand,
    )
    synthetic_lanes = max(
        1,
        int(np.ceil(desired_capacity / DEFAULT_SIMULATOR_CAPACITY_PER_LANE)),
    )
    synthetic_capacity = synthetic_lanes * DEFAULT_SIMULATOR_CAPACITY_PER_LANE
    maxmph = float(link_record.get('maxmph', 25.0) or 25.0)
    if not np.isfinite(maxmph) or maxmph <= 0:
        maxmph = 25.0
    length = float(synthetic_length_m)
    if not np.isfinite(length) or length <= 0:
        raise ValueError('synthetic_length_m must be positive and finite')

    def node_row(node_id, lon, lat, label):
        row = {column: np.nan for column in nodes.columns}
        row.update({
            'node_id': int(node_id),
            'lon': float(lon),
            'lat': float(lat),
            'node_osmid': f'synthetic_{label}_{target_id}',
            'type': 'synthetic_boundary',
        })
        return row

    def edge_row(link_id, start_id, end_id, start_lon_, start_lat_, end_lon_, end_lat_, label):
        row = {column: np.nan for column in edges.columns}
        row.update({
            'link_id': int(link_id),
            'start_node_id': int(start_id),
            'end_node_id': int(end_id),
            'type': 'synthetic_boundary',
            'length': length,
            'maxmph': maxmph,
            'lanes': int(synthetic_lanes),
            'capacity': float(synthetic_capacity),
            'start_osmid': f'synthetic_{label}_start_{target_id}',
            'end_osmid': f'synthetic_{label}_end_{target_id}',
            'edge_key': f'synthetic_{label}_{target_id}',
            'geometry': (
                f'LINESTRING ({start_lon_} {start_lat_}, '
                f'{end_lon_} {end_lat_})'
            ),
        })
        return row

    node_rows = []
    edge_rows = []
    synthetic_ids = []
    if predecessor is None:
        node_rows.append(node_row(source_node_id, source_lon, source_lat, 'source'))
        edge_rows.append(edge_row(
            source_link_id, source_node_id, int(link_record['start_node_id']),
            source_lon, source_lat, start_lon, start_lat, 'source',
        ))
        synthetic_ids.append(source_link_id)
    if successor is None:
        node_rows.append(node_row(sink_node_id, sink_lon, sink_lat, 'sink'))
        edge_rows.append(edge_row(
            sink_link_id, int(link_record['end_node_id']), sink_node_id,
            end_lon, end_lat, sink_lon, sink_lat, 'sink',
        ))
        synthetic_ids.append(sink_link_id)

    overlay_nodes = pd.concat([nodes, pd.DataFrame(node_rows)], ignore_index=True)
    overlay_edges = pd.concat([edges, pd.DataFrame(edge_rows)], ignore_index=True)
    overlay_nodes = overlay_nodes.sort_values('node_id', kind='mergesort').reset_index(drop=True)
    overlay_edges = overlay_edges.sort_values('link_id', kind='mergesort').reset_index(drop=True)
    os.makedirs(output_dir, exist_ok=True)
    overlay_nodes_csv = os.path.join(output_dir, 'nodes_with_synthetic_context.csv')
    overlay_edges_csv = os.path.join(output_dir, 'edges_with_synthetic_context.csv')
    overlay_nodes.to_csv(overlay_nodes_csv, index=False)
    overlay_edges.to_csv(overlay_edges_csv, index=False)

    predecessor = (
        overlay_edges.loc[overlay_edges['link_id'].astype(int) == source_link_id].iloc[0]
        if predecessor is None else predecessor
    )
    successor = (
        overlay_edges.loc[overlay_edges['link_id'].astype(int) == sink_link_id].iloc[0]
        if successor is None else successor
    )
    return overlay_nodes_csv, overlay_edges_csv, predecessor, successor, {
        'context_mode': 'synthetic_boundary',
        'synthetic_link_ids': [int(value) for value in synthetic_ids],
        'missing_context': missing,
        'synthetic_capacity': float(synthetic_capacity),
        'synthetic_lanes': int(synthetic_lanes),
        'synthetic_length_m': length,
    }


def _bpr_link_worker(job):
    """Run one complete flow sweep for one link in an isolated directory."""
    (link_record, nodes_csv, edges_csv, flow_spec, output_root, seed) = job
    link_id = int(link_record['link_id'])
    link_dir = os.path.join(output_root, f'link_{link_id}')
    os.makedirs(link_dir, exist_ok=True)
    started = time.perf_counter()
    result = {
        'link_id': link_id,
        'x_vector': [],
        'y_vector': [],
        'observations': [],
        'errors': [],
    }
    try:
        checkpoint_path = os.path.join(link_dir, "result.json")
        if isinstance(flow_spec, dict) and flow_spec.get("resume", False) and os.path.exists(checkpoint_path):
            with open(checkpoint_path) as handle:
                cached = json.load(handle)
            if cached.get("complete") and int(cached.get("link_id", -1)) == link_id:
                cached["resumed"] = True
                return cached
        historical_mode = (
            isinstance(flow_spec, dict)
            and flow_spec.get('mode') == 'historical_artifact_compatible'
        )
        if historical_mode:
            flow_levels = [float(value) for value in flow_spec['flow_levels']]
            flow_plan = [
                {'fraction': None, 'flow_rate': flow, 'demand_count': int(flow)}
                for flow in flow_levels
            ]
            result['flow_plan'] = flow_plan
            result['bpr_mode'] = 'historical_artifact_compatible'
        elif isinstance(flow_spec, dict):
            fractions = list(flow_spec['flow_fractions'])
            capacity_rate, flow_plan = _capacity_fraction_flow_plan(
                link_record,
                fractions,
                capacity_source=flow_spec.get('capacity_source', 'simulator'),
                capacity_per_lane=flow_spec.get('capacity_per_lane', DEFAULT_SIMULATOR_CAPACITY_PER_LANE),
                calibration_window_hours=flow_spec.get('calibration_window_hours', 0.1),
            )
            result['capacity_rate'] = capacity_rate
            result['flow_plan'] = flow_plan
        else:
            # Backward-compatible worker contract for small tests and legacy
            # callers that already provide absolute simulator counts.
            flow_plan = [
                {'fraction': None, 'flow_rate': float(flow), 'demand_count': int(flow)}
                for flow in flow_spec
            ]
        route_mode = 'link_probe' if isinstance(flow_spec, dict) else 'isolated_link'
        if isinstance(flow_spec, dict):
            route_mode = flow_spec.get('route_mode', route_mode)
        if route_mode not in {'link_probe', 'isolated_link', 'contextual'}:
            raise ValueError("route_mode must be 'link_probe', 'isolated_link', or 'contextual'")

        initial_od = os.path.join(link_dir, 'initial_od.csv')
        _write_single_od(
            initial_od,
            int(link_record['start_node_id']),
            int(link_record['end_node_id']),
            1,
        )
        finder = Runner(
            nodes_csv=nodes_csv,
            links_csv=edges_csv,
            od_csv=initial_od,
            seed=SeedManager(seed).derive('bpr-link', link_id),
        )
        sa_il = finder.find_sa_in_link(pd.Series(link_record))
        sa_ol = finder.find_sa_out_link(pd.Series(link_record))
        context_policy = (
            flow_spec.get('missing_context_policy', 'synthetic_boundary')
            if isinstance(flow_spec, dict) else 'proxy'
        )
        missing_context = route_mode == 'contextual' and (sa_il is None or sa_ol is None)
        if missing_context and context_policy == 'synthetic_boundary':
            (
                context_nodes_csv,
                context_edges_csv,
                sa_il,
                sa_ol,
                context_metadata,
            ) = _synthetic_boundary_overlay(
                nodes_csv,
                edges_csv,
                link_record,
                sa_il,
                sa_ol,
                flow_plan,
                link_dir,
                capacity_multiplier=(
                    flow_spec.get('synthetic_context_capacity_multiplier', 10.0)
                    if isinstance(flow_spec, dict) else 10.0
                ),
                synthetic_length_m=(
                    flow_spec.get('synthetic_context_length_m', 1.0)
                    if isinstance(flow_spec, dict) else 1.0
                ),
            )
            # Recreate the Runner from the worker-local overlay.  The
            # canonical files remain untouched and define all downstream
            # network identity.
            finder = Runner(
                nodes_csv=context_nodes_csv,
                links_csv=context_edges_csv,
                od_csv=initial_od,
                seed=SeedManager(seed).derive('bpr-link', link_id),
            )
            result.update(context_metadata)
            result['observation_source'] = 'simulated_synthetic_context'
        elif missing_context:
            if context_policy == 'fail_fast':
                result['fatal_context'] = True
            missing = []
            if sa_il is None:
                missing.append('straight-ahead predecessor')
            if sa_ol is None:
                missing.append('straight-ahead successor')
            raise RuntimeError(
                f"link {link_id} has no {' or '.join(missing)} for "
                f"contextual BPR measurement (policy={context_policy})"
            )
        else:
            result.update({
                'context_mode': 'contextual' if route_mode == 'contextual' else route_mode,
                'synthetic_link_ids': [],
                'missing_context': [],
                'observation_source': 'simulated_contextual',
            })
        if route_mode == 'link_probe':
            sa_ol = _probe_continuation(finder, link_record)
            if not sa_ol:
                raise RuntimeError(
                    f"link {link_id} has no outgoing continuation for link-level BPR probe"
                )
            _configure_probe_capacity(finder, link_record, sa_ol)
        if route_mode == 'contextual' and sa_il is not None and sa_ol is not None:
            link_orig = int(sa_il['start_node_id'])
            link_dest = int(sa_ol['end_node_id'])
        elif route_mode == 'link_probe':
            link_orig = int(link_record['start_node_id'])
            link_dest = int(sa_ol[-1]['end_node_id'])
        else:
            link_orig = int(link_record['start_node_id'])
            link_dest = int(link_record['end_node_id'])
        # Reuse the Runner and immutable network tables for the complete link
        # sweep. Each configured flow level is simulated exactly once.
        for flow_index, flow_item in enumerate(flow_plan):
            flow_rate = float(flow_item['flow_rate'])
            demand_count = int(flow_item['demand_count'])
            flow_dir = os.path.join(link_dir, f'flow_{flow_index}')
            os.makedirs(flow_dir, exist_ok=True)
            if demand_count == 0:
                # A zero-demand queue run creates no target-link observation.
                # The BPR baseline is exactly the canonical link free-flow
                # time, so record it directly as the 0.0C observation.
                result['x_vector'].append(flow_rate)
                result['y_vector'].append(_free_flow_time(link_record))
                result['observations'].append({
                    'fraction': flow_item.get('fraction'),
                    'requested_flow': flow_rate,
                    'realized_flow': 0.0,
                    'demand_count': 0,
                    'entries': 0,
                    'completions': 0,
                    'travel_time': _free_flow_time(link_record),
                    'travel_time_mean': _free_flow_time(link_record),
                    'travel_time_std': 0.0,
                    'travel_time_ci_half_width': 0.0,
                    'realized_flow_mean': 0.0,
                    'replications': 1,
                    'split': flow_item.get('split', 'training'),
                    'status': 'free_flow_anchor',
                })
                continue
            rep = 0
            rep_dir = flow_dir
            od_csv = os.path.join(rep_dir, 'od.csv')
            departure_times = (
                [0] * demand_count if historical_mode else
                _departure_schedule(
                    demand_count,
                    flow_spec.get('calibration_window_hours', 0.1)
                    if isinstance(flow_spec, dict) else 0.1,
                )
            )
            _write_single_od(
                od_csv, link_orig, link_dest, demand_count,
                departure_times=departure_times,
            )
            finder.od_df = pd.read_csv(od_csv)
            flow_seed = SeedManager(seed).derive(
                'bpr-flow', link_id, flow_index, demand_count, rep
            )
            finder.seed = flow_seed
            SeedManager(flow_seed)
            finder.add_charging_info(0, 0)
            route_sa_il = sa_il if route_mode == 'contextual' else None
            route_sa_ol = sa_ol if route_mode in {'contextual', 'link_probe'} else None
            finder.init_sq_simulation_for_bpr_function_fitting_V2(
                pd.Series(link_record), route_sa_il, route_sa_ol,
            )
            finder.spatial_queue_simulation(
                f'bpr_link_{link_id}_flow_{flow_index}_rep_{rep}',
                output_dir=rep_dir,
                t_end=int(flow_spec.get('simulation_horizon', 10801))
                if isinstance(flow_spec, dict) else 10801,
            )
            target_link = finder.sim.all_links.get(link_id)
            if target_link is None or not target_link.completed_travel_time_list:
                raise RuntimeError('target link did not produce a travel time')
            entries = int(target_link.tot_entering_vehs)
            completions = int(len(target_link.completed_travel_time_list))
            if entries != demand_count or completions != demand_count:
                raise RuntimeError(
                    f'target flow accounting mismatch: requested_agents={demand_count}, '
                    f'entries={entries}, completions={completions}'
                )
            if historical_mode:
                traffic_df = finder.return_traffic_data(
                    demand_count, link_orig, link_dest, density=0
                )
                target_rows = traffic_df.loc[
                    traffic_df['link_id'].astype(int) == link_id
                ]
                if target_rows.empty or not np.isfinite(float(target_rows.iloc[0]['flow'])):
                    raise RuntimeError(
                        f'link {link_id} did not produce a finite measured target flow'
                    )
                realized_flow = float(target_rows.iloc[0]['flow'])
            else:
                calibration_window = float(
                    flow_spec.get('calibration_window_hours', 0.1)
                    if isinstance(flow_spec, dict) else 0.1
                )
                realized_flow = (
                    entries / calibration_window
                    if isinstance(flow_spec, dict) else flow_rate
                )
            travel_time = float(target_link.ave_travel_time)
            realized_flow = float(realized_flow)
            result['x_vector'].append(realized_flow)
            result['y_vector'].append(travel_time)
            result['observations'].append({
                'fraction': flow_item.get('fraction'),
                'split': flow_item.get('split', 'training'),
                'requested_flow': flow_rate,
                'realized_flow': realized_flow,
                'realized_flow_mean': realized_flow,
                'demand_count': demand_count,
                'entries': entries,
                'completions': completions,
                'travel_time': travel_time,
                'travel_time_mean': travel_time,
                'travel_time_std': 0.0,
                'travel_time_ci_half_width': 0.0,
                'relative_ci_half_width': 0.0,
                'replications': 1,
                'target_queue_max': int(max(
                    [0] + [len(target_link.queue_veh), len(target_link.run_veh)]
                )),
                'status': 'simulated_measured_flow' if historical_mode else 'simulated',
                'continuation_link_ids': [int(item['link_id']) for item in sa_ol]
                if route_mode == 'link_probe' else [],
            })
    except Exception as exc:
        result['errors'].append(str(exc))
        result['traceback'] = traceback.format_exc()
    result['elapsed_seconds'] = time.perf_counter() - started
    result['complete'] = not result['errors']
    if result['complete']:
        atomic_write_json(checkpoint_path, result)
    return result


def _proxy_row(link, flow_levels, mode='capacity_fraction_strict', reason='', network_hash=None):
    length = float(link.get('length', 1) or 1)
    maxmph = float(link.get('maxmph', 25) or 25)
    # mph -> m/s is division by 2.2369.  Keep proxy values physically
    # dimensioned even though they remain degraded/non-simulated data.
    fft = length / (maxmph / 2.2369) if maxmph > 0 else 1.0
    return {
        'link_id': int(link['link_id']),
        'x_vector': [float(value) for value in flow_levels],
        'y_vector': [float(fft) for _ in flow_levels],
        'fit_status': 'proxy',
        'observation_source': 'proxy',
        'context_mode': 'proxy',
        'synthetic_link_ids': [],
        'missing_context': [],
        'bpr_mode': mode,
        'fallback_reason': reason,
        'sample_count': len(flow_levels),
        'network_hash': network_hash,
    }


def generate_bpr_data(coordinates=None, num_samples=25, max_flow=250,
                      work_dir=None, highway_types=None, road_net=None,
                      artifact_dir=None, workers=None, failure_policy='fail_fast',
                      allow_proxy=False, seed=0, timeout=None,
                      flow_fractions=None, capacity_source='simulator',
                      capacity_per_lane=DEFAULT_SIMULATOR_CAPACITY_PER_LANE,
                      calibration_window_hours=0.1, route_mode='link_probe',
                      mode='historical_artifact_compatible',
                      missing_context_policy='synthetic_boundary',
                      synthetic_context_capacity_multiplier=10.0,
                      synthetic_context_length_m=1.0,
                      simulation_horizon=10801, active_link_ids=None,
                      resume=True):
    """Generate BPR observations from one canonical artifact.

    Work is parallelized by link.  Each worker owns the complete flow sweep
    for its link and writes only below ``link_<id>``.
    """
    if not QUEUE_SIM_AVAILABLE and not (
        mode == 'historical_artifact_compatible' and
        (failure_policy == 'proxy' or allow_proxy)
    ):
        raise RuntimeError('Queue simulation not available')
    if failure_policy not in {'fail_fast', 'record', 'proxy'}:
        raise ValueError('failure_policy must be fail_fast, record, or proxy')
    if missing_context_policy not in {'synthetic_boundary', 'proxy', 'fail_fast'}:
        raise ValueError(
            'missing_context_policy must be synthetic_boundary, proxy, or fail_fast'
        )
    if float(synthetic_context_capacity_multiplier) <= 0:
        raise ValueError('synthetic_context_capacity_multiplier must be positive')
    if float(synthetic_context_length_m) <= 0:
        raise ValueError('synthetic_context_length_m must be positive')
    if int(simulation_horizon) < 1:
        raise ValueError('simulation_horizon must be >= 1')
    if mode not in {'historical_artifact_compatible', 'capacity_fraction_strict'}:
        raise ValueError('unsupported BPR generation mode')
    if work_dir is None:
        work_dir = os.getcwd()
    os.makedirs(work_dir, exist_ok=True)

    if artifact_dir is None:
        if road_net is None:
            if coordinates is None:
                raise ValueError('coordinates or a canonical artifact is required')
            road_net = RoadNet('generated')
            road_net.get_map(*coordinates, highway_types=highway_types)
        artifact_dir = os.path.join(work_dir, 'network_artifact')
        road_net.export_artifact(artifact_dir, source={'coordinates': coordinates, 'highway_types': highway_types})

    nodes, edges, manifest = load_network_artifact(artifact_dir)
    # Pass the artifact's exact files to every worker; do not serialize a
    # second topology copy that could drift from the validated hash.
    nodes_csv = os.path.join(artifact_dir, manifest['nodes_file'])
    edges_csv = os.path.join(artifact_dir, manifest['edges_file'])
    if mode == 'historical_artifact_compatible':
        flow_levels = np.linspace(1, max_flow, num_samples).astype(int).tolist()
        flow_spec = {
            'mode': mode,
            'flow_levels': flow_levels,
            'route_mode': 'contextual',
            'missing_context_policy': missing_context_policy,
            'synthetic_context_capacity_multiplier': float(synthetic_context_capacity_multiplier),
            'synthetic_context_length_m': float(synthetic_context_length_m),
            'simulation_horizon': int(simulation_horizon),
        }
        legacy_flow_levels = flow_levels
        flow_fractions = None
    elif flow_fractions is None:
        flow_fractions = list(DEFAULT_FLOW_FRACTIONS)
        flow_fractions = [float(value) for value in flow_fractions]
        flow_spec = {
            'mode': mode,
            'flow_fractions': flow_fractions,
            'capacity_source': capacity_source,
            'capacity_per_lane': float(capacity_per_lane),
            'calibration_window_hours': float(calibration_window_hours),
            'route_mode': route_mode,
            'missing_context_policy': missing_context_policy,
            'synthetic_context_capacity_multiplier': float(synthetic_context_capacity_multiplier),
            'synthetic_context_length_m': float(synthetic_context_length_m),
            'simulation_horizon': int(simulation_horizon),
            'resume': bool(resume),
        }
        legacy_flow_levels = None
    elif flow_fractions is not None:
        flow_fractions = [float(value) for value in flow_fractions]
        flow_spec = {
            'mode': mode,
            'flow_fractions': flow_fractions,
            'capacity_source': capacity_source,
            'capacity_per_lane': float(capacity_per_lane),
            'calibration_window_hours': float(calibration_window_hours),
            'route_mode': route_mode,
            'missing_context_policy': missing_context_policy,
            'synthetic_context_capacity_multiplier': float(synthetic_context_capacity_multiplier),
            'synthetic_context_length_m': float(synthetic_context_length_m),
            'simulation_horizon': int(simulation_horizon),
            'resume': bool(resume),
        }
        legacy_flow_levels = None
    sweep_name = (
        'link_sweeps_historical'
        if mode == 'historical_artifact_compatible'
        else ('link_sweeps_capacity_fraction' if flow_fractions is not None else 'link_sweeps')
    )
    output_root = os.path.join(work_dir, sweep_name)
    os.makedirs(output_root, exist_ok=True)
    link_records = edges.sort_values(
        ['lanes', 'link_id'], ascending=[False, True], kind='mergesort'
    ).to_dict(orient='records')
    active_ids = None if active_link_ids is None else {int(value) for value in active_link_ids}
    jobs = [] if not QUEUE_SIM_AVAILABLE else [
        (record, nodes_csv, edges_csv, flow_spec, output_root,
         SeedManager(seed).derive('bpr-job', int(record['link_id'])))
        for record in link_records
        if int(record['start_node_id']) != int(record['end_node_id'])
        and (active_ids is None or int(record['link_id']) in active_ids)
    ]
    results = []
    pool_size = 0
    if jobs:
        pool_size = workers if workers is not None else min(len(jobs), available_cpus())
        with Pool(max(1, int(pool_size))) as pool:
            # Unordered consumption prevents a slow early link from blocking
            # completed checkpoints and progress from all other workers.
            results.extend(pool.imap_unordered(_bpr_link_worker, jobs, chunksize=1))

    by_id = {int(result['link_id']): result for result in results}
    rows = []
    failures = []
    for record in link_records:
        link_id = int(record['link_id'])
        if active_ids is not None and link_id not in active_ids:
            continue
        result = by_id.get(link_id)
        if mode == 'historical_artifact_compatible':
            expected_samples = len(flow_spec['flow_levels'])
        else:
            expected_samples = len(flow_spec) if not isinstance(flow_spec, dict) else len(flow_spec['flow_fractions'])
        if result and len(result['x_vector']) == expected_samples:
            row = {
                'link_id': link_id,
                'x_vector': result['x_vector'],
                'y_vector': result['y_vector'],
                'calibration_capacity': result.get('capacity_rate'),
                'calibration_fft': _free_flow_time(record),
                'calibration_window_hours': float(calibration_window_hours) if flow_fractions is not None else None,
                'fit_status': 'simulated',
                'observation_source': result.get('observation_source', 'simulated_contextual'),
                'context_mode': result.get('context_mode', route_mode),
                'synthetic_link_ids': json.dumps(result.get('synthetic_link_ids', [])),
                'missing_context': json.dumps(result.get('missing_context', [])),
                'bpr_mode': mode,
                'sample_count': expected_samples,
                'fallback_reason': '',
                'network_hash': manifest['network_hash'],
                'observations_json': json.dumps(result.get('observations', [])),
            }
            rows.append(row)
        else:
            result = by_id.get(link_id, {})
            failures.append({
                'link_id': link_id,
                'errors': result.get('errors', ['worker was not scheduled or returned no result']),
                'traceback': result.get('traceback'),
                'observed_samples': len(result.get('x_vector', [])),
                'expected_samples': expected_samples,
                'fatal_context': bool(result.get('fatal_context', False)),
            })
            if (failure_policy == 'proxy' or allow_proxy) and not result.get('fatal_context', False):
                if mode == 'historical_artifact_compatible':
                    rows.append(_proxy_row(
                        record, flow_spec['flow_levels'], mode=mode,
                        reason='; '.join(result.get('errors', ['simulation unavailable'])),
                        network_hash=manifest['network_hash'],
                    ))
                elif flow_fractions is not None:
                    _, plan = _capacity_fraction_flow_plan(
                        record, flow_fractions,
                        capacity_source=capacity_source,
                        capacity_per_lane=capacity_per_lane,
                        calibration_window_hours=calibration_window_hours,
                    )
                    rows.append(_proxy_row(record, [item['flow_rate'] for item in plan], mode=mode,
                                           reason='; '.join(result.get('errors', ['simulation unavailable'])),
                                           network_hash=manifest['network_hash']))
                else:
                    rows.append(_proxy_row(record, legacy_flow_levels, mode=mode,
                                           reason='; '.join(result.get('errors', ['simulation unavailable'])),
                                           network_hash=manifest['network_hash']))

    fatal_failures = bool(failures) and (
        any(item.get('fatal_context', False) for item in failures)
        or (failure_policy == 'fail_fast' and not allow_proxy)
    )
    df = pd.DataFrame(rows).sort_values('link_id').reset_index(drop=True)
    observation_rows = []
    for result in results:
        for observation in result.get('observations', []):
            observation_rows.append({'link_id': int(result['link_id']), **observation})
    if observation_rows:
        pd.DataFrame(observation_rows).to_csv(
            os.path.join(work_dir, 'bpr_observations.csv.gz'), index=False,
            compression='gzip',
        )
    atomic_write_json(os.path.join(work_dir, 'bpr_manifest.json'), {
            'network_hash': manifest['network_hash'],
            'bpr_mode': mode,
            'num_samples': int(num_samples),
            'max_flow': float(max_flow),
            'random_seed': int(seed if seed is not None else 0),
            'fitter_version': (
                'historical_v1' if mode == 'historical_artifact_compatible'
                else 'capacity_fraction_v1'
            ),
            'historical_reference_commit': '37eab33' if mode == 'historical_artifact_compatible' else None,
            'route_semantics': 'measured_target_flow_with_straight_ahead_context' if mode == 'historical_artifact_compatible' else route_mode,
            'link_count': len(link_records),
            'successful_links': int(len(df)),
            'failures': failures,
            'failure_policy': failure_policy,
            'missing_context_policy': (
                flow_spec.get('missing_context_policy', 'synthetic_boundary')
                if isinstance(flow_spec, dict) else None
            ),
            'synthetic_context_capacity_multiplier': (
                float(flow_spec.get('synthetic_context_capacity_multiplier', 10.0))
                if isinstance(flow_spec, dict) else None
            ),
            'synthetic_context_length_m': (
                float(flow_spec.get('synthetic_context_length_m', 1.0))
                if isinstance(flow_spec, dict) else None
            ),
            'simulation_horizon': int(
                flow_spec.get('simulation_horizon', 10801)
                if isinstance(flow_spec, dict) else 10801
            ),
            'observation_source_counts': (
                df.get('observation_source', pd.Series(dtype=str)).value_counts().to_dict()
            ),
            'synthetic_context_links': [
                int(row['link_id']) for _, row in df.iterrows()
                if row.get('observation_source') == 'simulated_synthetic_context'
            ],
            'workers': pool_size,
            'workers_available': available_cpus(),
            'timeout': timeout,
            'flow_levels': legacy_flow_levels,
            'flow_fractions': flow_fractions,
            'fit_status_counts': (
                df.get('fit_status', pd.Series(dtype=str)).value_counts().to_dict()
            ),
            'capacity_source': capacity_source if flow_fractions is not None else None,
            'capacity_per_lane': float(capacity_per_lane) if flow_fractions is not None else None,
            'calibration_window_hours': float(calibration_window_hours) if flow_fractions is not None else None,
            'route_mode': 'contextual' if mode == 'historical_artifact_compatible' else (route_mode if flow_fractions is not None else None),
            'flow_unit': 'simulator_measured_target_flow' if mode == 'historical_artifact_compatible' else ('vehicles_per_hour_equivalent' if flow_fractions is not None else 'simulator_agent_count'),
            'flow_levels_by_link': {
                str(int(result['link_id'])): {
                    'capacity_rate': result.get('capacity_rate'),
                    'flow_rates': [item['flow_rate'] for item in result.get('flow_plan', [])],
                    'demand_counts': [item['demand_count'] for item in result.get('flow_plan', [])],
                }
                for result in results if 'flow_plan' in result
            },
            'link_timings': [
                {
                    'link_id': int(result['link_id']),
                    'elapsed_seconds': float(result.get('elapsed_seconds', 0.0)),
                    'sample_count': len(result.get('x_vector', [])),
                    'observation_source': result.get('observation_source'),
                    'context_mode': result.get('context_mode'),
                    'synthetic_link_ids': result.get('synthetic_link_ids', []),
                    'missing_context': result.get('missing_context', []),
                    'observations': result.get('observations', []),
                    'errors': result.get('errors', []),
                    'fatal_context': bool(result.get('fatal_context', False)),
                    'traceback': result.get('traceback'),
                }
                for result in sorted(results, key=lambda value: int(value['link_id']))
            ],
        })
    if fatal_failures:
        raise RuntimeError(
            f'BPR generation failed for {len(failures)} links; first failure: {failures[0]}'
        )
    return df, len(link_records if active_ids is None else active_ids)


def generate_and_save_bpr_data(coordinates, output_path, **kwargs):
    df, n_links = generate_bpr_data(coordinates, **kwargs)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    meta_path = output_path.replace('.csv', '_meta.json')
    with open(meta_path, 'w') as handle:
        json.dump({'n_links': n_links}, handle)
    print(f'BPR data saved to {output_path}')
    return df, n_links


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate BPR data')
    parser.add_argument('--coordinates', type=str, required=True)
    parser.add_argument('--output', type=str, default='data/traffic_data.csv')
    parser.add_argument('--num-samples', type=int, default=25)
    parser.add_argument('--max-flow', type=int, default=250)
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    coords = [float(value) for value in args.coordinates.split(',')]
    generate_and_save_bpr_data(
        coords, args.output, num_samples=args.num_samples,
        max_flow=args.max_flow, workers=args.workers,
    )
