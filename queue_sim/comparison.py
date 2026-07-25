"""Greedy vs exhaustive charger placement comparison via queue simulation.

Adapted from run_comparison.py with:
- Config-driven parameters
- Single-swap refinement (ported from CG outer_optimization)
- Fixed seeding per (rep, placement) to eliminate MC noise asymmetry
"""
import json
import os
import pickle
import random
import warnings
from itertools import combinations
from multiprocessing import Pool

import numpy as np

warnings.filterwarnings('ignore')

from queue_sim import Runner, QUEUE_SIM_AVAILABLE


def _collapse_repeats(lst):
    if not lst:
        return []
    out = [lst[0]]
    for x in lst[1:]:
        if x != out[-1]:
            out.append(x)
    return out


def _prune_flow_data(data, charger_locs_tuple, k):
    flow_data = {'no charging type': [], 'charging type': []}
    for r in data['configurations'][charger_locs_tuple]['reconstruction_results']['k_metrics'][k]['routes']:
        if r['type'] == 'non_charging':
            flow_data['no charging type'].append({'path': r['links'], 'flow': r['flow'], 'station node': None})
        else:
            flow_data['charging type'].append({'path': _collapse_repeats(r['links']), 'flow': r['flow'], 'station node': r['charger']})
    return flow_data


def _placement_seed(rep, positions):
    return rep * 100000 + sum(sorted(set(positions)))


def _run_sim(positions, data, ne, k, orig, dest, per_ev, per_ch, name, work_dir,
             ent_cap, ch_cap, ex_cap, cost, seed=None):
    locs = tuple(sorted(set(positions)))
    loc_str = ','.join(map(str, locs))
    if loc_str not in ne:
        return float('inf')
    flow_data = _prune_flow_data(data, locs, k)
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    try:
        r = Runner(
            nodes_csv=os.path.join(work_dir, f'traffic_inputs_{name}_nodes.csv'),
            links_csv=os.path.join(work_dir, f'traffic_inputs_{name}_edges.csv'),
            od_csv=os.path.join(work_dir, f'traffic_inputs_{name}_od.csv'),
        )
        r.add_charging_info(per_ev, per_ch)
        for pos in positions:
            r.create_EV_charging_station_at_node(pos, ent_cap, ch_cap, ex_cap, cost)
        r.init_sq_simulation_with_path_assignment({(orig, dest): flow_data}, ne[loc_str]['ch'], ne[loc_str]['no_ch'])
        r.spatial_queue_simulation(name, output_dir=os.path.join(work_dir, 'traffic_outputs'))
    except Exception:
        return float('inf')
    return r.tot_travel_time


def _greedy_rep(args):
    (rep, file_path, ne_path, k, num_stations, possible_positions,
     orig, dest, per_ev, per_ch, name, work_dir,
     ent_cap, ch_cap, ex_cap, cost, single_swap) = args

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    with open(ne_path, 'rb') as f:
        ne = pickle.load(f)

    best_positions = []
    best_time = float('inf')
    remaining = list(possible_positions)

    for _ in range(num_stations):
        best_round_time = float('inf')
        new_best = None
        for candid in remaining:
            positions = [candid] + best_positions
            tt = _run_sim(positions, data, ne, k, orig, dest, per_ev, per_ch, name, work_dir,
                          ent_cap, ch_cap, ex_cap, cost, seed=_placement_seed(rep, positions))
            if tt < best_round_time:
                best_round_time = tt
                new_best = candid
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
                for i, sel in enumerate(best_positions):
                    trial = list(best_positions)
                    trial[i] = unsel
                    tt = _run_sim(trial, data, ne, k, orig, dest, per_ev, per_ch, name, work_dir,
                                  ent_cap, ch_cap, ex_cap, cost, seed=_placement_seed(rep, trial))
                    if tt < best_time:
                        best_time = tt
                        best_positions = trial
                        improved = True
                        break

    return best_positions, best_time


def _exhaustive_rep(args):
    (rep, file_path, ne_path, k, combs,
     orig, dest, per_ev, per_ch, name, work_dir,
     ent_cap, ch_cap, ex_cap, cost) = args

    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    with open(ne_path, 'rb') as f:
        ne = pickle.load(f)

    return [
        _run_sim(list(comb), data, ne, k, orig, dest, per_ev, per_ch, name, work_dir,
                 ent_cap, ch_cap, ex_cap, cost, seed=_placement_seed(rep, list(comb)))
        for comb in combs
    ]


def run_comparison(config, experiment_dir, all_opt_results_path, ne_assignments_path,
                   network_name='college_park'):
    """Run greedy vs exhaustive comparison.

    Returns dict with best_greedy, best_exhaustive, suboptimality, all_results.
    """
    if not QUEUE_SIM_AVAILABLE:
        raise RuntimeError(f"Queue simulation not available: {__import__('queue_sim')._QUEUE_SIM_ERROR}")

    q = config.queue_simulation
    k = q['K']
    n_reps = q['N']
    num_stations = config.num_chargers
    possible_positions = config.possible_charger_positions
    workers = q['WORKERS']
    ent_cap = q['ENT_CAPACITY']
    ch_cap = q['CHARGING_CAPACITY']
    ex_cap = q['EXIT_CAPACITY']
    cost = q['COST']
    single_swap = q.get('single_swap', True)

    work_dir = os.path.join(experiment_dir, 'queue')

    with open(all_opt_results_path, 'rb') as f:
        data = pickle.load(f)

    od = data['run_configuration']['od_demand']
    orig, dest = next(iter(od))
    num_charging = od[(orig, dest)][1]
    num_not_charging = od[(orig, dest)][0]
    num_of_vehs = num_charging + num_not_charging
    per_ev = num_charging / num_of_vehs
    per_ch = 1

    print(f'Origin: {orig}  Destination: {dest}')
    print(f'Vehicles: {num_of_vehs}  ({num_charging} charging, {num_not_charging} non-charging)')
    print(f'Candidate positions: {possible_positions}  Placing: {num_stations} charger(s)')

    combs = [list(c) for c in combinations(possible_positions, num_stations)]
    sim_params = (orig, dest, per_ev, per_ch, network_name, work_dir, ent_cap, ch_cap, ex_cap, cost)

    print(f'\nRunning greedy  ({n_reps} reps, parallelized{" + swap" if single_swap else ""}) ...')
    greedy_args = [
        (rep, all_opt_results_path, ne_assignments_path, k, num_stations, possible_positions, *sim_params, single_swap)
        for rep in range(n_reps)
    ]
    with Pool(workers) as pool:
        greedy_raw = pool.map(_greedy_rep, greedy_args)

    pos_hist = [r[0] for r in greedy_raw]
    time_hist = [r[1] for r in greedy_raw]

    unique_pos = []
    for pos in pos_hist:
        if pos not in unique_pos and list(reversed(pos)) not in unique_pos:
            unique_pos.append(pos)

    greedy_avg_tt = []
    for pos in unique_pos:
        times = [time_hist[i] for i, p in enumerate(pos_hist)
                 if p == pos or p == list(reversed(pos))]
        greedy_avg_tt.append(sum(times) / len(times))

    print(f'Running exhaustive  ({n_reps} reps x {len(combs)} combinations, parallelized) ...')
    exhaustive_args = [
        (rep, all_opt_results_path, ne_assignments_path, k, combs, *sim_params)
        for rep in range(n_reps)
    ]
    with Pool(workers) as pool:
        exhaustive_raw = pool.map(_exhaustive_rep, exhaustive_args)

    exhaustive_avg_tt = np.mean(exhaustive_raw, axis=0).tolist()

    print('\n-- Greedy --------------------------------------------------')
    for pos, tt in zip(unique_pos, greedy_avg_tt):
        print(f'  {pos}  ->  avg travel time = {tt:.1f}')

    print('\n-- Exhaustive ----------------------------------------------')
    for comb, tt in zip(combs, exhaustive_avg_tt):
        print(f'  {comb}  ->  avg travel time = {tt:.1f}')

    best_g_tt = min(greedy_avg_tt)
    best_g_pos = unique_pos[greedy_avg_tt.index(best_g_tt)]
    best_e_idx = int(np.argmin(exhaustive_avg_tt))
    best_e_tt = exhaustive_avg_tt[best_e_idx]
    best_e_pos = combs[best_e_idx]
    subopt = (best_g_tt - best_e_tt) / best_e_tt * 100 if best_e_tt > 0 else 0.0

    print('\n-- Summary -------------------------------------------------')
    print(f'  Best greedy     : {best_g_pos}  (avg tt = {best_g_tt:.1f})')
    print(f'  Best exhaustive : {best_e_pos}  (avg tt = {best_e_tt:.1f})')
    print(f'  Greedy suboptimality: {subopt:.2f}%')

    results = {
        'best_greedy': {'positions': best_g_pos, 'avg_travel_time': best_g_tt},
        'best_exhaustive': {'positions': best_e_pos, 'avg_travel_time': best_e_tt},
        'suboptimality_pct': subopt,
        'greedy_results': [{'positions': p, 'avg_travel_time': t} for p, t in zip(unique_pos, greedy_avg_tt)],
        'exhaustive_results': [{'positions': c, 'avg_travel_time': t} for c, t in zip(combs, exhaustive_avg_tt)],
        'config': {'N': n_reps, 'K': k, 'num_stations': num_stations, 'single_swap': single_swap},
    }

    results_path = os.path.join(work_dir, 'comparison_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {results_path}')

    return results
