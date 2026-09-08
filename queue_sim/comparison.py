"""Greedy versus exhaustive charger-placement comparison."""

from __future__ import annotations

import json
import os
import pickle
import random
import time
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

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
    seed = seed_manager.seed if seed_manager is not None else config.pipeline.get('random_seed', 0)
    num_stations = config.num_chargers
    possible_positions = config.possible_charger_positions
    combinations_list = enumerate_placements(possible_positions, num_stations)
    required_placements = set(combinations_list) | set(
        enumerate_placements(possible_positions, 1)
    )
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
