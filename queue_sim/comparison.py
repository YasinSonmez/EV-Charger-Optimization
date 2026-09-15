"""Greedy versus exhaustive charger-placement comparison."""

from __future__ import annotations

import json
import os
import pickle
import random
import time
import hashlib
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings('ignore')

from queue_sim import Runner, QUEUE_SIM_AVAILABLE
from queue_sim.find_nash import CYCLE_APPROXIMATION_STATUS, _prune_flow_data
from src.contracts import (
    SeedManager,
    canonical_placement,
    enumerate_placements,
    normalize_od_demand,
    ordered_unique_positions,
    single_swap_neighbors,
)
from src.network_artifact import load_network_artifact
from src.run_state import available_cpus


def _placement_seed(rep, positions):
    return rep * 100000 + sum(sorted(set(positions)))


def _assignment_is_usable(value):
    """Accept exact NE or explicitly labeled retained cycle approximations."""
    if not isinstance(value, dict) or value.get('status') == 'failed':
        return False
    return bool(value.get('converged', False)) or (
        value.get('status') == CYCLE_APPROXIMATION_STATUS
        and bool(value.get('approximate_equilibrium', False))
    )


def _run_sim(positions, data, ne, k, demand_classes, input_paths, output_root,
             ent_cap, ch_cap, ex_cap, cost, simulation_horizon,
             seed=None, scenario='placement'):
    locs = canonical_placement(positions)
    loc_str = ','.join(map(str, locs))
    if loc_str not in ne:
        raise KeyError(f'No Nash assignment for charger placement {loc_str}')
    flow_data = ne[loc_str].get('flow_data') or _prune_flow_data(data, locs, k)
    assignments = ne[loc_str].get('assignments', {})
    assignments_ch = assignments.get('F2', assignments.get('ch', {}))
    assignments_no = assignments.get('F1', assignments.get('no_ch', {}))
    nodes_path, edges_path, od_path = input_paths
    runner = Runner(
        nodes_csv=nodes_path,
        links_csv=edges_path,
        od_csv=od_path,
        seed=seed,
    )
    for pos in positions:
        runner.create_EV_charging_station_at_node(pos, ent_cap, ch_cap, ex_cap, cost)
    runner.init_sq_simulation_with_path_assignment(
        flow_data, assignments_ch, assignments_no,
    )
    output_dir = os.path.join(output_root, 'traffic_outputs', scenario, loc_str)
    os.makedirs(output_dir, exist_ok=True)
    runner.spatial_queue_simulation(
        f'{scenario}_{loc_str}',
        t_end=int(simulation_horizon),
        output_dir=output_dir,
    )
    return float(runner.tot_travel_time)


def _comparison_rep(args):
    """Run one paired replication, caching each unordered placement once."""
    (rep, file_path, ne_path, k, num_stations, possible_positions,
     demand_classes, input_paths, work_dir, ent_cap, ch_cap, ex_cap, cost,
     simulation_horizon, single_swap, combinations_list, seed) = args
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle)
    with open(ne_path, 'rb') as handle:
        ne = pickle.load(handle)

    cache = {}
    cache_elapsed = {}
    cache_hits = 0

    def evaluate(positions):
        nonlocal cache_hits
        placement = canonical_placement(positions)
        if placement in cache:
            cache_hits += 1
            return cache[placement]
        started = time.perf_counter()
        cache[placement] = _run_sim(
            placement, data, ne, k, demand_classes, input_paths, work_dir,
            ent_cap, ch_cap, ex_cap, cost, simulation_horizon,
            seed=SeedManager(seed).derive('placement', rep, placement),
            scenario=f'comparison_rep_{rep}',
        )
        cache_elapsed[placement] = time.perf_counter() - started
        return cache[placement]

    # Every replication must observe the same placement universe. Previously
    # each replication followed its own noisy greedy prefix, leaving some
    # intermediate placements without all paired observations and causing the
    # mean-search summary to fail after simulation had already completed.
    for size in range(1, int(num_stations) + 1):
        for placement in enumerate_placements(possible_positions, size):
            evaluate(placement)

    best_positions = []
    best_time = float('inf')
    remaining = list(ordered_unique_positions(possible_positions))
    for _ in range(num_stations):
        best_round_time = float('inf')
        new_best = None
        for candidate in remaining:
            positions = canonical_placement([candidate] + best_positions)
            value = evaluate(positions)
            if value < best_round_time:
                best_round_time = value
                new_best = candidate
        if new_best is None:
            raise RuntimeError(f'No feasible greedy candidate at round {len(best_positions)}')
        best_positions.append(new_best)
        remaining.remove(new_best)
        best_time = best_round_time

    if single_swap:
        swap_results = [
            (evaluate(trial), trial)
            for trial in single_swap_neighbors(best_positions, possible_positions)
        ]
        if swap_results:
            swap_time, swap_positions = min(
                swap_results, key=lambda value: (value[0], value[1])
            )
            if swap_time < best_time:
                best_time = swap_time
                best_positions = list(swap_positions)

    exhaustive_values = [evaluate(combination) for combination in combinations_list]
    return {
        'greedy_positions': canonical_placement(best_positions),
        'greedy_time': float(best_time),
        'exhaustive_values': exhaustive_values,
        'placement_values': cache,
        'placement_elapsed_seconds': cache_elapsed,
        'unique_simulations': len(cache),
        'cache_hits': cache_hits,
    }


def _build_queue_search_summary(
    paired_results, possible_positions, num_stations, single_swap,
):
    """Reconstruct one mean-objective search path from paired replications."""
    candidates = ordered_unique_positions(possible_positions)
    sample_count = len(paired_results)
    samples = {}
    elapsed = {}
    for result in paired_results:
        for placement, value in result['placement_values'].items():
            placement = canonical_placement(placement)
            samples.setdefault(placement, []).append(float(value))
        for placement, value in result['placement_elapsed_seconds'].items():
            placement = canonical_placement(placement)
            elapsed[placement] = elapsed.get(placement, 0.0) + float(value)
    objective = {
        placement: float(np.mean(values))
        for placement, values in samples.items()
        if len(values) == sample_count
    }

    trace = []
    trace_by_placement = {}

    def record(placement, phase, round_index=None):
        placement = canonical_placement(placement)
        if placement not in objective:
            raise RuntimeError(
                'Mean queue search is missing complete paired observations for '
                f'placement {placement}'
            )
        if placement in trace_by_placement:
            trace_by_placement[placement]['also_used_by'].append(phase)
            return
        item = {
            'evaluation_order': len(trace) + 1,
            'phase': phase,
            'round': round_index,
            'placement': list(placement),
            'objective': objective[placement],
            'worker_seconds': float(elapsed.get(placement, 0.0)),
            'also_used_by': [],
        }
        trace.append(item)
        trace_by_placement[placement] = item

    selected = ()
    greedy_rounds = []
    for round_index in range(1, int(num_stations) + 1):
        trials = [
            canonical_placement(selected + (candidate,))
            for candidate in candidates if candidate not in selected
        ]
        for placement in trials:
            record(placement, 'greedy', round_index)
        selected = min(trials, key=lambda value: objective[value])
        greedy_rounds.append({
            'round': round_index,
            'trials': [list(value) for value in trials],
            'selected': list(selected),
            'objective': objective[selected],
        })
    greedy_selection = selected

    swap_selection = greedy_selection
    if single_swap:
        swap_trials = single_swap_neighbors(greedy_selection, candidates)
        for placement in swap_trials:
            record(placement, 'single_swap')
        swap_selection = min(
            [greedy_selection] + swap_trials,
            key=lambda value: objective[value],
        )

    exhaustive_trials = enumerate_placements(candidates, num_stations)
    for placement in exhaustive_trials:
        record(placement, 'exhaustive')
    exhaustive_selection = min(
        exhaustive_trials, key=lambda value: objective[value]
    )

    phase_worker_seconds = {
        phase: float(sum(
            item['worker_seconds'] for item in trace if item['phase'] == phase
        ))
        for phase in ('greedy', 'single_swap', 'exhaustive')
    }
    return {
        'candidate_order': list(candidates),
        'num_chargers': int(num_stations),
        'single_swap_enabled': bool(single_swap),
        'greedy': {
            'placement': list(greedy_selection),
            'objective': objective[greedy_selection],
        },
        'single_swap': {
            'placement': list(swap_selection),
            'objective': objective[swap_selection],
        },
        'exhaustive': {
            'placement': list(exhaustive_selection),
            'objective': objective[exhaustive_selection],
        },
        'greedy_rounds': greedy_rounds,
        'trace': trace,
        'phase_worker_seconds': phase_worker_seconds,
        'paired_replications': sample_count,
    }


def _correlation_summary(cg_values, queue_values):
    """Pearson and Spearman correlations for paired finite placements."""
    x = np.asarray(cg_values, dtype=float)
    y = np.asarray(queue_values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    result = {
        'paired_count': int(x.size),
        'paired_placements': int(x.size),
        'pearson': None,
        'spearman': None,
    }
    if x.size < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return result
    result['pearson'] = float(np.corrcoef(x, y)[0, 1])
    x_rank = pd.Series(x).rank(method='average').to_numpy()
    y_rank = pd.Series(y).rank(method='average').to_numpy()
    result['spearman'] = float(np.corrcoef(x_rank, y_rank)[0, 1])
    return result


def _reviewer_baselines(config, experiment_dir, artifact_dir, cg_objectives,
                        queue_objectives, cg_search, queue_search):
    """Extract placement baselines from the already evaluated universe."""
    candidates = tuple(int(value) for value in config.possible_charger_positions)
    count = int(config.num_chargers)
    resolved_path = os.path.join(experiment_dir, 'resolved_config.json')
    metadata = {}
    if os.path.isfile(resolved_path):
        with open(resolved_path) as handle:
            metadata = json.load(handle).get('generated_scenario') or {}
    detour = {
        int(item['node_id']): float(item.get('minimum_detour_ratio', float('inf')))
        for item in metadata.get('candidates', [])
    }
    minimum_detour = canonical_placement(sorted(
        candidates, key=lambda node: (detour.get(node, float('inf')), node)
    )[:count])

    _, edges, _ = load_network_artifact(artifact_dir)
    graph = nx.DiGraph()
    for row in edges.itertuples():
        weight = float(getattr(row, 'travel_time', getattr(row, 'length', 1.0)))
        old = graph.get_edge_data(int(row.start_node_id), int(row.end_node_id))
        if old is None or weight < old['weight']:
            graph.add_edge(int(row.start_node_id), int(row.end_node_id), weight=weight)
    centrality = nx.betweenness_centrality(graph, weight='weight', normalized=True)
    betweenness = canonical_placement(sorted(
        candidates, key=lambda node: (-centrality.get(node, 0.0), node)
    )[:count])

    def summarize(objectives, model_search):
        optimum = min(objectives.values())
        placements = {
            'greedy': canonical_placement(model_search['greedy']['placement']),
            'single_swap': canonical_placement(model_search['single_swap']['placement']),
            'exhaustive': min(objectives, key=objectives.get),
            'minimum_detour': minimum_detour,
            'weighted_betweenness': betweenness,
        }
        output = {}
        for method, placement in placements.items():
            objective = float(objectives[placement])
            output[method] = {
                'placement': list(placement), 'objective': objective,
                'regret_pct': 100.0 * (objective - optimum) / optimum,
            }
        random_objective = float(np.mean(list(objectives.values())))
        output['uniform_random_expectation'] = {
            'placement': None, 'objective': random_objective,
            'regret_pct': 100.0 * (random_objective - optimum) / optimum,
        }
        return output

    return {
        'congestion_game': summarize(cg_objectives, cg_search),
        'queue': summarize(queue_objectives, queue_search),
    }


def run_comparison(config, experiment_dir, all_opt_results_path, ne_assignments_path,
                   network_name='canonical', artifact_dir=None, seed_manager=None):
    """Compare greedy and exhaustive placement using all OD/type demand."""
    started = time.perf_counter()
    if not QUEUE_SIM_AVAILABLE:
        raise RuntimeError(f"Queue simulation not available: {__import__('queue_sim')._QUEUE_SIM_ERROR}")
    if artifact_dir is None:
        raise ValueError('A canonical network artifact is required for queue comparison')
    _, _, network_manifest = load_network_artifact(artifact_dir)

    q = config.queue_simulation
    k = q['K']
    n_reps = q['N']
    workers_requested = q['WORKERS']
    workers = workers_requested
    if workers is None:
        workers = config.pipeline.get('parallel_workers')
    available_workers = available_cpus()
    workers = available_workers if workers is None else max(1, min(int(workers), available_workers))
    work_dir = os.path.join(experiment_dir, 'queue')
    input_paths = (
        os.path.join(artifact_dir, network_manifest['nodes_file']),
        os.path.join(artifact_dir, network_manifest['edges_file']),
        os.path.join(work_dir, 'canonical_od.csv'),
    )
    if not all(os.path.exists(path) for path in input_paths):
        raise FileNotFoundError('Queue input files were not created by find_nash_assignments')

    with open(all_opt_results_path, 'rb') as handle:
        data = pickle.load(handle)
    with open(ne_assignments_path, 'rb') as handle:
        ne = pickle.load(handle)
    ne_hashes = {
        value.get('network_hash')
        for value in ne.values()
        if isinstance(value, dict) and value.get('network_hash')
    }
    if ne_hashes and ne_hashes != {network_manifest['network_hash']}:
        raise ValueError(
            f'Queue assignments use network hashes {sorted(ne_hashes)}, '
            f'but the requested artifact is {network_manifest["network_hash"]}'
        )
    queue_manifest_path = os.path.join(work_dir, 'queue_manifest.json')
    with open(queue_manifest_path) as handle:
        queue_manifest = json.load(handle)
    with open(input_paths[2], 'rb') as handle:
        actual_od_checksum = hashlib.sha256(handle.read()).hexdigest()
    if queue_manifest.get('canonical_od_checksum') != actual_od_checksum:
        raise ValueError('Canonical OD schedule checksum changed after NE assignment')
    identities = {
        json.dumps(value.get('queue_identity'), sort_keys=True)
        for value in ne.values() if isinstance(value, dict)
    }
    expected_identity = json.dumps(queue_manifest.get('queue_identity'), sort_keys=True)
    if identities != {expected_identity}:
        raise ValueError('Queue assignment identity does not match queue manifest')
    invalid = {
        key: value.get('status', 'invalid')
        for key, value in ne.items()
        if not _assignment_is_usable(value)
    }
    if invalid:
        raise RuntimeError(
            'Queue comparison requires exact equilibria or explicitly labeled '
            f'cycle-state approximations: {invalid}'
        )
    approximate = sorted(
        key for key, value in ne.items()
        if value.get('status') == CYCLE_APPROXIMATION_STATUS
    )
    if approximate:
        print(
            'WARNING: queue comparison is using current assignments retained '
            f'at cycle detection for {len(approximate)} configurations. '
            'These are approximations, not verified Nash equilibria.'
        )
    demand_classes = normalize_od_demand(data['run_configuration']['od_demand'])
    pipeline_seed = seed_manager.seed if seed_manager is not None else config.pipeline.get('random_seed', 0)
    seed = int(q.get('seed') if q.get('seed') is not None else pipeline_seed)
    num_stations = config.num_chargers
    possible_positions = config.possible_charger_positions
    combinations_list = enumerate_placements(possible_positions, num_stations)
    required_placements = {
        placement
        for size in range(1, int(num_stations) + 1)
        for placement in enumerate_placements(possible_positions, size)
    }
    cg_placements = {
        canonical_placement(value) for value in data.get('configurations', {})
    }
    missing_cg = sorted(required_placements - cg_placements)
    missing_ne = sorted(
        placement for placement in required_placements
        if ','.join(map(str, placement)) not in ne
    )
    if missing_cg or missing_ne:
        raise ValueError(
            'Queue comparison placement universe does not match CG/NE outputs: '
            f'missing_from_cg={missing_cg}, missing_from_ne={missing_ne}'
        )
    sim_args = (
        demand_classes, input_paths, work_dir, q['ENT_CAPACITY'],
        q['CHARGING_CAPACITY'], q['EXIT_CAPACITY'], q['COST'],
        q.get('SIMULATION_HORIZON', 10801),
    )

    comparison_args = [
        (rep, all_opt_results_path, ne_assignments_path, k, num_stations,
         possible_positions, *sim_args, q.get('single_swap', True),
         combinations_list, seed)
        for rep in range(n_reps)
    ]
    with Pool(workers) as pool:
        paired_raw = pool.map(_comparison_rep, comparison_args)

    exhaustive_values = np.asarray(
        [value['exhaustive_values'] for value in paired_raw], dtype=float
    )
    exhaustive_avg = np.mean(exhaustive_values, axis=0).tolist()
    search = _build_queue_search_summary(
        paired_raw, possible_positions, num_stations,
        q.get('single_swap', True),
    )
    post_greedy = search['single_swap'] if q.get('single_swap', True) else search['greedy']
    greedy_results = [{
        'positions': post_greedy['placement'],
        'avg_travel_time': float(post_greedy['objective']),
    }]
    exhaustive_results = [
        {'positions': list(combination), 'avg_travel_time': float(value)}
        for combination, value in zip(combinations_list, exhaustive_avg)
    ]
    cg_objectives = {
        canonical_placement(placement): float(value['objective_value'])
        for placement, value in data.get('configurations', {}).items()
        if len(canonical_placement(placement)) == int(num_stations)
        and np.isfinite(float(value.get('objective_value', np.nan)))
    }
    queue_objectives = {
        canonical_placement(value['positions']): float(value['avg_travel_time'])
        for value in exhaustive_results
    }
    paired_placements = [
        placement for placement in combinations_list
        if placement in cg_objectives and placement in queue_objectives
    ]
    correlations = _correlation_summary(
        [cg_objectives[value] for value in paired_placements],
        [queue_objectives[value] for value in paired_placements],
    )
    correlations['placements'] = [list(value) for value in paired_placements]
    cg_ranked = sorted(paired_placements, key=lambda value: (cg_objectives[value], value))
    queue_ranked = sorted(paired_placements, key=lambda value: (queue_objectives[value], value))
    top_count = min(3, len(paired_placements))
    correlations.update({
        'top_1_agreement': bool(cg_ranked and queue_ranked and cg_ranked[0] == queue_ranked[0]),
        'top_3_overlap_count': len(set(cg_ranked[:top_count]) & set(queue_ranked[:top_count])),
        'top_3_denominator': top_count,
    })
    cg_search = data.get('placement_search', {})
    reviewer_baselines = (
        _reviewer_baselines(
            config, experiment_dir, artifact_dir, cg_objectives,
            queue_objectives, cg_search, search,
        )
        if cg_search else {}
    )

    ne_statistics = {}
    if os.path.isfile(queue_manifest_path):
        with open(queue_manifest_path) as handle:
            queue_manifest = json.load(handle)
        ne_statistics = {
            'status_counts': queue_manifest.get('status_counts', {}),
            'iteration_statistics': queue_manifest.get('iteration_statistics', {}),
            'cycle_length_statistics': queue_manifest.get(
                'cycle_length_statistics', {}
            ),
            'final_gap_statistics': queue_manifest.get('final_gap_statistics', {}),
            'minimum_gap_statistics': queue_manifest.get('minimum_gap_statistics', {}),
            'final_gap_mean_normalized_statistics': queue_manifest.get(
                'final_gap_mean_normalized_statistics', {}
            ),
            'minimum_gap_mean_normalized_statistics': queue_manifest.get(
                'minimum_gap_mean_normalized_statistics', {}
            ),
            'configuration_statuses': queue_manifest.get(
                'configuration_statuses', {}
            ),
        }

    best_greedy = greedy_results[0]
    best_exhaustive = min(exhaustive_results, key=lambda value: value['avg_travel_time'])
    best_e_time = best_exhaustive['avg_travel_time']
    suboptimality = (
        (best_greedy['avg_travel_time'] - best_e_time) / best_e_time * 100
        if best_e_time > 0 else 0.0
    )
    results = {
        'status': (
            'complete_with_approximate_cycle_states'
            if approximate else 'complete'
        ),
        'uses_approximate_ne': bool(approximate),
        'exact_ne_eligible': not approximate,
        'approximate_configurations': approximate,
        'assignment_quality': (
            'current_state_at_cycle_detection'
            if approximate else 'converged_nash'
        ),
        'best_greedy': best_greedy,
        'greedy_before_swap': {
            'positions': search['greedy']['placement'],
            'avg_travel_time': search['greedy']['objective'],
        },
        'best_single_swap': {
            'positions': search['single_swap']['placement'],
            'avg_travel_time': search['single_swap']['objective'],
        },
        'best_exhaustive': best_exhaustive,
        'suboptimality_pct': float(suboptimality),
        'greedy_results': greedy_results,
        'exhaustive_results': exhaustive_results,
        'placement_search': search,
        'ne_statistics': ne_statistics,
        'cg_queue_correlations': correlations,
        'reviewer_baselines': reviewer_baselines,
        'reviewer_baselines': reviewer_baselines,
        'config': {
            'N': n_reps, 'K': k, 'num_stations': num_stations,
            'single_swap': q.get('single_swap', True),
            'greedy_method': (
                'greedy_plus_single_swap'
                if q.get('single_swap', True) else 'greedy'
            ),
            'multi_od': True,
        },
        'network_hash': network_manifest['network_hash'],
        'timing': {
            'elapsed_seconds': time.perf_counter() - started,
            'workers': int(workers),
            'workers_requested': workers_requested,
            'workers_available': available_workers,
            'replications': int(n_reps),
            'unique_simulations': int(sum(
                value['unique_simulations'] for value in paired_raw
            )),
            'placement_cache_hits': int(sum(
                value['cache_hits'] for value in paired_raw
            )),
            'paired_placement_cache': True,
        },
    }
    result_path = os.path.join(work_dir, 'comparison_results.json')
    with open(result_path, 'w') as handle:
        json.dump(results, handle, indent=2)
    print(f'Results saved to {result_path}')
    return results
