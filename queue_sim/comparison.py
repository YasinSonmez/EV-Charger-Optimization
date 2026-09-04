"""Greedy versus exhaustive charger-placement comparison."""

from __future__ import annotations

import json
import os
import pickle
import random
import time
import warnings
from itertools import combinations
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from queue_sim import Runner, QUEUE_SIM_AVAILABLE
from queue_sim.find_nash import _prune_flow_data
from src.contracts import normalize_od_demand, SeedManager
from src.network_artifact import load_network_artifact
from src.run_state import available_cpus


def _placement_seed(rep, positions):
    return rep * 100000 + sum(sorted(set(positions)))


def _run_sim(positions, data, ne, k, demand_classes, input_paths, output_root,
             ent_cap, ch_cap, ex_cap, cost, simulation_horizon,
             seed=None, scenario='placement'):
    locs = tuple(sorted(set(positions)))
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


def _greedy_rep(args):
    (rep, file_path, ne_path, k, num_stations, possible_positions,
     demand_classes, input_paths, work_dir, ent_cap, ch_cap, ex_cap, cost,
     simulation_horizon, single_swap, seed) = args
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle)
    with open(ne_path, 'rb') as handle:
        ne = pickle.load(handle)

    best_positions = []
    best_time = float('inf')
    remaining = list(possible_positions)
    for _ in range(num_stations):
        best_round_time = float('inf')
        new_best = None
        for candidate in remaining:
            positions = [candidate] + best_positions
            value = _run_sim(
                positions, data, ne, k, demand_classes, input_paths, work_dir,
                ent_cap, ch_cap, ex_cap, cost,
                simulation_horizon,
                seed=SeedManager(seed).derive('placement', rep, positions),
                scenario=f'greedy_rep_{rep}',
            )
            if value < best_round_time:
                best_round_time = value
                new_best = candidate
        if new_best is None:
            raise RuntimeError(f'No feasible greedy candidate at round {len(best_positions)}')
        best_positions.append(new_best)
        remaining.remove(new_best)
        best_time = best_round_time

    if single_swap:
        improved = True
        while improved:
            improved = False
            for unsel in possible_positions:
                if unsel in best_positions:
                    continue
                for index, _selected in enumerate(best_positions):
                    trial = list(best_positions)
                    trial[index] = unsel
                    value = _run_sim(
                        trial, data, ne, k, demand_classes, input_paths, work_dir,
                        ent_cap, ch_cap, ex_cap, cost,
                        simulation_horizon,
                        seed=SeedManager(seed).derive('placement', rep, trial),
                        scenario=f'greedy_swap_rep_{rep}',
                    )
                    if value < best_time:
                        best_time = value
                        best_positions = trial
                        improved = True
                        break
                if improved:
                    break
    return best_positions, best_time


def _exhaustive_rep(args):
    (rep, file_path, ne_path, k, combs, demand_classes, input_paths,
     work_dir, ent_cap, ch_cap, ex_cap, cost, simulation_horizon, seed) = args
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle)
    with open(ne_path, 'rb') as handle:
        ne = pickle.load(handle)
    values = []
    for combination in combs:
        values.append(_run_sim(
            list(combination), data, ne, k, demand_classes, input_paths, work_dir,
            ent_cap, ch_cap, ex_cap, cost,
            simulation_horizon,
            seed=SeedManager(seed).derive('placement', rep, combination),
            scenario=f'exhaustive_rep_{rep}',
        ))
    return values


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
        if not isinstance(value, dict) or not value.get('converged', False)
    }
    if invalid:
        raise RuntimeError(
            f'Queue comparison requires converged Nash assignments: {invalid}'
        )
    demand_classes = normalize_od_demand(data['run_configuration']['od_demand'])
    seed = seed_manager.seed if seed_manager is not None else config.pipeline.get('random_seed', 0)
    num_stations = config.num_chargers
    possible_positions = config.possible_charger_positions
    combinations_list = [list(c) for c in combinations(possible_positions, num_stations)]
    sim_args = (
        demand_classes, input_paths, work_dir, q['ENT_CAPACITY'],
        q['CHARGING_CAPACITY'], q['EXIT_CAPACITY'], q['COST'],
        q.get('SIMULATION_HORIZON', 10801),
    )

    greedy_args = [
        (rep, all_opt_results_path, ne_assignments_path, k, num_stations,
         possible_positions, *sim_args, q.get('single_swap', True), seed)
        for rep in range(n_reps)
    ]
    with Pool(workers) as pool:
        greedy_raw = pool.map(_greedy_rep, greedy_args)

    position_history = [value[0] for value in greedy_raw]
    time_history = [value[1] for value in greedy_raw]
    unique_positions = []
    for positions in position_history:
        normalized = sorted(positions)
        if normalized not in unique_positions:
            unique_positions.append(normalized)
    greedy_results = []
    for positions in unique_positions:
        values = [time_history[i] for i, observed in enumerate(position_history)
                  if sorted(observed) == positions]
        greedy_results.append({'positions': positions, 'avg_travel_time': float(np.mean(values))})

    exhaustive_args = [
        (rep, all_opt_results_path, ne_assignments_path, k, combinations_list,
         *sim_args, seed)
        for rep in range(n_reps)
    ]
    with Pool(workers) as pool:
        exhaustive_raw = pool.map(_exhaustive_rep, exhaustive_args)
    exhaustive_values = np.asarray(exhaustive_raw, dtype=float)
    exhaustive_avg = np.mean(exhaustive_values, axis=0).tolist()
    exhaustive_results = [
        {'positions': combination, 'avg_travel_time': float(value)}
        for combination, value in zip(combinations_list, exhaustive_avg)
    ]

    best_greedy = min(greedy_results, key=lambda value: value['avg_travel_time'])
    best_exhaustive = min(exhaustive_results, key=lambda value: value['avg_travel_time'])
    best_e_time = best_exhaustive['avg_travel_time']
    suboptimality = (
        (best_greedy['avg_travel_time'] - best_e_time) / best_e_time * 100
        if best_e_time > 0 else 0.0
    )
    results = {
        'best_greedy': best_greedy,
        'best_exhaustive': best_exhaustive,
        'suboptimality_pct': float(suboptimality),
        'greedy_results': greedy_results,
        'exhaustive_results': exhaustive_results,
        'config': {
            'N': n_reps, 'K': k, 'num_stations': num_stations,
            'single_swap': q.get('single_swap', True),
            'multi_od': True,
        },
        'network_hash': network_manifest['network_hash'],
        'timing': {
            'elapsed_seconds': time.perf_counter() - started,
            'workers': int(workers),
            'workers_requested': workers_requested,
            'workers_available': available_workers,
            'replications': int(n_reps),
        },
    }
    result_path = os.path.join(work_dir, 'comparison_results.json')
    with open(result_path, 'w') as handle:
        json.dump(results, handle, indent=2)
    print(f'Results saved to {result_path}')
    return results
