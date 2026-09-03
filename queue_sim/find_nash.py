"""Multi-OD better-response Nash assignments for the queue simulator."""

from __future__ import annotations

import os
import pickle
import time
import warnings
import json
from multiprocessing import Pool

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from queue_sim import Runner, QUEUE_SIM_AVAILABLE
from src.contracts import DemandClass, SeedManager, normalize_od_demand
from src.network_artifact import load_network_artifact


def _collapse_repeats(lst):
    if not lst:
        return []
    out = [lst[0]]
    for x in lst[1:]:
        if x != out[-1]:
            out.append(x)
    return out


def _collapse_path_and_links(path, link_ids):
    """Collapse repeated node markers while keeping link IDs aligned."""
    if not path:
        return [], []
    compact_path = [path[0]]
    compact_links = []
    for index in range(len(path) - 1):
        next_node = path[index + 1]
        link_id = link_ids[index] if index < len(link_ids) else None
        if next_node == compact_path[-1]:
            continue
        compact_links.append(link_id)
        compact_path.append(next_node)
    return compact_path, compact_links


def _od_key(value):
    if isinstance(value, str):
        origin, destination = value.split(',')
        return int(origin), int(destination)
    return tuple(int(x) for x in value)


def _prune_flow_data(data, charger_locs_tuple, k):
    """Extract route-flow records for every OD pair in one CG result."""
    flow_data = {}
    config = data['configurations'][charger_locs_tuple]
    metrics = config['reconstruction_results']['k_metrics'][k]
    for route_index, route in enumerate(metrics['routes']):
        od = (int(route['origin']), int(route['destination']))
        group = flow_data.setdefault(od, {'no charging type': [], 'charging type': []})
        if route['type'] == 'non_charging':
            group['no charging type'].append({
                'route_id': route.get('route_id', f'{od[0]}_{od[1]}_F1_none_{route_index}'),
                'path': route['links'],
                'link_ids': route.get('link_ids', []),
                'flow': route['flow'],
                'station node': None,
            })
        else:
            path, link_ids = _collapse_path_and_links(
                route['links'], route.get('link_ids', [])
            )
            group['charging type'].append({
                'route_id': route.get(
                    'route_id',
                    f"{od[0]}_{od[1]}_F2_{route.get('charger')}_{route_index}",
                ),
                'path': path,
                'link_ids': link_ids,
                'flow': route['flow'],
                'station node': route.get('charger'),
                'station_cost': route.get('station_cost', 0.0),
            })
    return flow_data


def _rounded_counts(routes, total):
    raw = [max(0.0, float(r.get('flow', 0.0))) for r in routes]
    if not routes or total <= 0:
        return [0] * len(routes)
    route_total = sum(raw)
    if route_total <= 0:
        counts = [0] * len(routes)
        counts[0] = int(total)
        return counts
    quotas = [value / route_total * total for value in raw]
    counts = [int(np.floor(value)) for value in quotas]
    remainder = int(total) - sum(counts)
    order = sorted(range(len(counts)), key=lambda i: (-(quotas[i] - counts[i]), i))
    for index in order[:max(0, remainder)]:
        counts[index] += 1
    return counts


def _write_queue_inputs(artifact_dir, work_dir, demand_classes):
    nodes, edges, manifest = load_network_artifact(artifact_dir)
    os.makedirs(work_dir, exist_ok=True)
    nodes_path = os.path.join(work_dir, 'canonical_nodes.csv')
    edges_path = os.path.join(work_dir, 'canonical_edges.csv')
    od_path = os.path.join(work_dir, 'canonical_od.csv')
    nodes.to_csv(nodes_path, index=False)
    edges.to_csv(edges_path, index=False)
    rows = []
    for record in demand_classes:
        for _ in range(record.demand):
            rows.append({
                'origin_node_id': record.origin,
                'destin_node_id': record.destination,
                'origin_osmid': record.origin,
                'destin_osmid': record.destination,
                'is_EV': record.vehicle_type == 'F2',
                'need_to_charge': record.vehicle_type == 'F2',
                'current_charge': 0,
                'target_charge': 100,
                'go_to_station_id': np.nan,
            })
    pd.DataFrame(rows).to_csv(od_path, index=False)
    return (
        os.path.join(artifact_dir, manifest['nodes_file']),
        os.path.join(artifact_dir, manifest['edges_file']),
        od_path,
        manifest,
    )


def _simulate(args):
    (charger_locs_tuple, flow_data, assignments_ch, assignments_no,
     num_iters, input_paths, output_root, ent_cap, ch_cap, ex_cap, cost,
     simulation_horizon, seed, scenario_name) = args
    nodes_path, edges_path, od_path = input_paths
    route_samples = []
    total_times = []
    for rep in range(int(num_iters)):
        rep_dir = os.path.join(output_root, f'rep_{rep}')
        os.makedirs(rep_dir, exist_ok=True)
        rep_seed = SeedManager(seed).derive('queue-rep', scenario_name, rep)
        runner = Runner(
            nodes_csv=nodes_path,
            links_csv=edges_path,
            od_csv=od_path,
            seed=rep_seed,
        )
        for loc in charger_locs_tuple:
            runner.create_EV_charging_station_at_node(loc, ent_cap, ch_cap, ex_cap, cost)
        runner.init_sq_simulation_with_path_assignment(
            flow_data, assignments_ch, assignments_no,
        )
        runner.spatial_queue_simulation(
            scenario_name,
            t_end=int(simulation_horizon),
            output_dir=rep_dir,
        )
        route_samples.append(runner.check_NE(return_details=True))
        total_times.append(float(runner.tot_travel_time))

    averaged = {}
    for sample in route_samples:
        for group_key, routes in sample.items():
            target = averaged.setdefault(group_key, [])
            while len(target) < len(routes):
                target.append({'travel_time': [], 'agent_count': 0, 'used': False})
            for index, route in enumerate(routes):
                target[index]['travel_time'].append(float(route['travel_time']))
                target[index]['agent_count'] = route['agent_count']
                target[index]['used'] = route['used']
    for routes in averaged.values():
        for route in routes:
            values = route.pop('travel_time')
            route['travel_time'] = float(np.mean(values)) if values else float('inf')
    return averaged, float(np.mean(total_times)) if total_times else float('inf')


def _simulate_rep(args):
    """Run one queue replication for one Nash iteration.

    Replications are deliberately independent.  Keeping this as a top-level
    worker function lets the parent process distribute the actual expensive
    simulator calls across the configured process pool instead of assigning
    one serial replication loop to each charger configuration.
    """
    (charger_locs_tuple, flow_data, assignments_ch, assignments_no,
     input_paths, output_root, ent_cap, ch_cap, ex_cap, cost,
     simulation_horizon, seed, scenario_name, rep) = args
    started = time.perf_counter()
    rep_dir = os.path.join(output_root, f'rep_{rep}')
    try:
        os.makedirs(rep_dir, exist_ok=True)
        rep_seed = SeedManager(seed).derive('queue-rep', scenario_name, rep)
        nodes_path, edges_path, od_path = input_paths
        runner = Runner(
            nodes_csv=nodes_path,
            links_csv=edges_path,
            od_csv=od_path,
            seed=rep_seed,
        )
        for loc in charger_locs_tuple:
            runner.create_EV_charging_station_at_node(loc, ent_cap, ch_cap, ex_cap, cost)
        runner.init_sq_simulation_with_path_assignment(
            flow_data, assignments_ch, assignments_no,
        )
        runner.spatial_queue_simulation(
            scenario_name,
            t_end=int(simulation_horizon),
            output_dir=rep_dir,
        )
        return {
            'rep': int(rep),
            'status': 'ok',
            'scenario_name': scenario_name,
            'details': runner.check_NE(return_details=True),
            'total_time': float(runner.tot_travel_time),
            'elapsed_seconds': time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            'rep': int(rep),
            'status': 'failed',
            'scenario_name': scenario_name,
            'failure_reason': repr(exc),
            'elapsed_seconds': time.perf_counter() - started,
        }


def _aggregate_simulation_samples(samples):
    """Aggregate independent replication outputs deterministically."""
    successful = [sample for sample in samples if sample.get('status') == 'ok']
    if not successful:
        return {}, float('inf')

    averaged = {}
    for sample in sorted(successful, key=lambda item: item['rep']):
        for group_key, routes in sample['details'].items():
            target = averaged.setdefault(group_key, [])
            while len(target) < len(routes):
                target.append({'travel_time': [], 'agent_count': 0, 'used': False})
            for index, route in enumerate(routes):
                target[index]['travel_time'].append(float(route['travel_time']))
                target[index]['agent_count'] = route['agent_count']
                target[index]['used'] = route['used']
    for routes in averaged.values():
        for route in routes:
            values = route.pop('travel_time')
            route['travel_time'] = float(np.mean(values)) if values else float('inf')

    total_time = float(np.mean([sample['total_time'] for sample in successful]))
    return averaged, total_time


def _relative_gap(details):
    best = None
    best_group = None
    for group_key, routes in details.items():
        if not routes:
            continue
        available = [float(route['travel_time']) for route in routes]
        used = [float(route['travel_time']) for route in routes if route.get('used')]
        if not used or not available:
            continue
        minimum = min(available)
        maximum_used = max(used)
        if minimum <= 0 or not np.isfinite(minimum):
            relative = float('inf') if maximum_used > minimum else 0.0
        else:
            relative = max(0.0, (maximum_used - minimum) / minimum)
        if best is None or relative > best:
            best = relative
            best_group = (group_key, routes, maximum_used, minimum)
    return (float(best or 0.0), best_group)


def _nash_for_config(args):
    (charger_locs_tuple, file_path, k, alpha, num_iters, max_ne_iterations,
     input_paths, work_dir, ent_cap, ch_cap, ex_cap, cost,
     simulation_horizon, seed) = args
    loc_str = ','.join(map(str, charger_locs_tuple))
    with open(file_path, 'rb') as handle:
        data = pickle.load(handle)
    flow_data = _prune_flow_data(data, charger_locs_tuple, k)
    demand_classes = normalize_od_demand(data['run_configuration']['od_demand'])
    demand_by_od_type = {
        ((record.origin, record.destination), record.vehicle_type): record.demand
        for record in demand_classes
    }
    assignments_ch = {}
    assignments_no = {}
    for od, group in flow_data.items():
        assignments_no[od] = _rounded_counts(
            group['no charging type'], demand_by_od_type.get((od, 'F1'), 0)
        )
        assignments_ch[od] = _rounded_counts(
            group['charging type'], demand_by_od_type.get((od, 'F2'), 0)
        )

    history = []
    iteration_timings = []
    converged = False
    final_details = {}
    for iteration in range(int(max_ne_iterations)):
        iteration_started = time.perf_counter()
        try:
            final_details, _ = _simulate((
                charger_locs_tuple, flow_data, assignments_ch, assignments_no,
                num_iters, input_paths, os.path.join(work_dir, 'traffic_outputs', loc_str,
                                                     f'ne_{iteration}'),
                ent_cap, ch_cap, ex_cap, cost, simulation_horizon,
                seed, f'ne_{loc_str}_{iteration}',
            ))
        except Exception as exc:
            iteration_timings.append({
                'iteration': iteration,
                'elapsed_seconds': time.perf_counter() - iteration_started,
                'status': 'failed',
            })
            return loc_str, {
                'status': 'failed',
                'failure_reason': str(exc),
                'assignments': {'F1': assignments_no, 'F2': assignments_ch},
                'converged': False,
                'iterations': iteration + 1,
                'final_gap': float('inf'),
                'route_metrics': {},
                'flow_data': flow_data,
                'iteration_timings': iteration_timings,
            }, history + [float('inf')]
        iteration_timings.append({
            'iteration': iteration,
            'elapsed_seconds': time.perf_counter() - iteration_started,
        'status': 'ok',
        })
        gap, selected = _relative_gap(final_details)
        history.append(gap)
        if gap <= float(alpha) or selected is None:
            converged = gap <= float(alpha)
            break
        group_key, routes, maximum_used, minimum = selected
        od, vehicle_type = group_key
        counts = assignments_ch if vehicle_type == 'F2' else assignments_no
        current = counts[od]
        used_indices = [i for i, route in enumerate(routes) if route.get('used') and current[i] > 0]
        min_index = min(range(len(routes)), key=lambda i: routes[i]['travel_time'])
        max_index = max(used_indices, key=lambda i: routes[i]['travel_time']) if used_indices else None
        if max_index is None or min_index == max_index:
            break
        current[max_index] -= 1
        current[min_index] += 1

    result = {
        'assignments': {
            'F1': assignments_no,
            'F2': assignments_ch,
        },
        'converged': converged,
        'iterations': len(history),
        'final_gap': history[-1] if history else float('inf'),
        'route_metrics': final_details,
        'flow_data': flow_data,
        'iteration_timings': iteration_timings,
        'route_assignments': [
            {
                'od_id': f'{od[0]}_{od[1]}',
                'vehicle_type': vehicle_type,
                'charger_id': entry.get('station node') if vehicle_type == 'F2' else None,
                'route_id': entry.get('route_id', f'{od[0]}_{od[1]}_{vehicle_type}_{index}'),
                'route_index': index,
                'count': int(count),
            }
            for od, group in flow_data.items()
            for vehicle_type, entries, counts in (
                ('F1', group['no charging type'], assignments_no.get(od, [])),
                ('F2', group['charging type'], assignments_ch.get(od, [])),
            )
            for index, (entry, count) in enumerate(zip(entries, counts))
        ],
    }
    return loc_str, result, history


def find_nash_assignments(config, experiment_dir, all_opt_results_path,
                          network_name='canonical', artifact_dir=None,
                          seed_manager=None):
    """Find a shared-network Nash assignment for every charger configuration.

    The simulator calls are the parallel work unit.  Nash iterations are
    sequential because each iteration depends on the previous assignment, but
    all independent ``(configuration, replication)`` jobs in an iteration are
    dispatched through one process pool.  This uses all configured workers,
    avoids nested process pools, and keeps every output directory unique.
    """
    if not QUEUE_SIM_AVAILABLE:
        raise RuntimeError(f"Queue simulation not available: {__import__('queue_sim')._QUEUE_SIM_ERROR}")

    q = config.queue_simulation
    alpha = q.get('ALPHA', 0.01)
    num_iters = q.get('NUM_ITERS', 1)
    max_ne_iterations = q.get('MAX_NE_ITERATIONS', 200)
    workers = q.get('WORKERS')
    if workers is None:
        workers = config.pipeline.get('parallel_workers')
    work_dir = os.path.join(experiment_dir, 'queue')
    os.makedirs(work_dir, exist_ok=True)
    for sub in ('t_stats', 'link_stats', 'node_stats'):
        os.makedirs(os.path.join(work_dir, 'traffic_outputs', sub), exist_ok=True)

    with open(all_opt_results_path, 'rb') as handle:
        data = pickle.load(handle)
    demand_classes = normalize_od_demand(data['run_configuration']['od_demand'])
    if artifact_dir is None:
        raise ValueError('A canonical network artifact is required for queue simulation')
    nodes_path, edges_path, od_path, manifest = _write_queue_inputs(
        artifact_dir, work_dir, demand_classes
    )
    with open(os.path.join(work_dir, 'network_manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2)

    input_paths = (nodes_path, edges_path, od_path)
    configs = list(data['configurations'].keys())
    seed = seed_manager.seed if seed_manager is not None else config.pipeline.get('random_seed', 0)
    available_workers = max(1, os.cpu_count() or 1)
    pool_size = available_workers if workers is None else max(1, int(workers))
    pool_size = min(pool_size, available_workers)

    with open(all_opt_results_path, 'rb') as handle:
        data = pickle.load(handle)
    demand_classes = normalize_od_demand(data['run_configuration']['od_demand'])
    demand_by_od_type = {
        ((record.origin, record.destination), record.vehicle_type): record.demand
        for record in demand_classes
    }

    states = {}
    resume_path = q.get('resume_from')
    resume_assignments = {}
    if resume_path:
        with open(resume_path, 'rb') as handle:
            resume_assignments = pickle.load(handle)
        if not isinstance(resume_assignments, dict):
            raise ValueError('queue_simulation.resume_from must contain a mapping')

    for charger_locs in configs:
        charger_locs_tuple = tuple(charger_locs)
        loc_str = ','.join(map(str, charger_locs_tuple))
        flow_data = _prune_flow_data(data, charger_locs_tuple, q['K'])
        assignments_no = {}
        assignments_ch = {}
        for od, group in flow_data.items():
            assignments_no[od] = _rounded_counts(
                group['no charging type'],
                demand_by_od_type.get((od, 'F1'), 0),
            )
            assignments_ch[od] = _rounded_counts(
                group['charging type'],
                demand_by_od_type.get((od, 'F2'), 0),
            )
        states[loc_str] = {
            'charger_locs': charger_locs_tuple,
            'flow_data': flow_data,
            'assignments_no': assignments_no,
            'assignments_ch': assignments_ch,
            'history': [],
            'iteration_timings': [],
            'final_details': {},
            'converged': False,
            'status': 'ok',
            'failure_reason': None,
            'resumed_complete': False,
        }

        prior = resume_assignments.get(loc_str)
        if (
            isinstance(prior, dict)
            and prior.get('status', 'ok') != 'failed'
            and int(prior.get('iterations', 0)) >= int(max_ne_iterations)
        ):
            prior_hash = prior.get('network_hash')
            if prior_hash and prior_hash != manifest['network_hash']:
                raise ValueError(
                    f'Resumed queue assignment {loc_str} uses network hash '
                    f'{prior_hash}, expected {manifest["network_hash"]}'
                )
            states[loc_str].update({
                'assignments_no': prior.get('assignments', {}).get('F1', assignments_no),
                'assignments_ch': prior.get('assignments', {}).get('F2', assignments_ch),
                'history': [float(prior.get('final_gap', float('inf')))],
                'iteration_timings': prior.get('iteration_timings', []),
                'final_details': prior.get('route_metrics', {}),
                'converged': bool(prior.get('converged', False)),
                'status': prior.get('status', 'ok'),
                'failure_reason': prior.get('failure_reason'),
                'resumed_complete': True,
            })

    active = {
        loc_str for loc_str, state in states.items()
        if not state.get('resumed_complete')
    }
    for iteration in range(int(max_ne_iterations)):
        if not active:
            break
        iteration_started = time.perf_counter()
        args_list = []
        for loc_str in sorted(active):
            state = states[loc_str]
            output_root = os.path.join(
                work_dir, 'traffic_outputs', loc_str, f'ne_{iteration}'
            )
            scenario_name = f'ne_{loc_str}_{iteration}'
            for rep in range(int(num_iters)):
                args_list.append((
                    state['charger_locs'], state['flow_data'],
                    state['assignments_ch'], state['assignments_no'],
                    input_paths, output_root, q['ENT_CAPACITY'],
                    q['CHARGING_CAPACITY'], q['EXIT_CAPACITY'], q['COST'],
                    q.get('SIMULATION_HORIZON', 10801), seed,
                    scenario_name, rep,
                ))

        with Pool(pool_size) as pool:
            raw_samples = pool.map(_simulate_rep, args_list)

        by_config = {loc_str: [] for loc_str in active}
        for sample in raw_samples:
            # The task list is constructed in configuration blocks.  Use the
            # output path-independent rep index and scenario convention to
            # associate results with the corresponding block.
            scenario = sample.get('scenario_name')
            if scenario is not None:
                loc_str = scenario.rsplit('_', 1)[0].removeprefix('ne_')
                if loc_str in by_config:
                    by_config[loc_str].append(sample)

        # Older Python workers do not need to return scenario_name: attach the
        # association deterministically from the task ordering as a fallback.
        if sum(len(value) for value in by_config.values()) != len(raw_samples):
            by_config = {loc_str: [] for loc_str in active}
            offset = 0
            for loc_str in sorted(active):
                by_config[loc_str] = raw_samples[offset:offset + int(num_iters)]
                offset += int(num_iters)

        iteration_elapsed = time.perf_counter() - iteration_started
        for loc_str in sorted(active):
            state = states[loc_str]
            samples = by_config[loc_str]
            failures = [sample for sample in samples if sample.get('status') != 'ok']
            state['iteration_timings'].append({
                'iteration': iteration,
                'elapsed_seconds': iteration_elapsed,
                'replications': int(num_iters),
                'successful_replications': int(len(samples) - len(failures)),
                'workers': pool_size,
                'status': 'failed' if failures else 'ok',
            })
            if failures:
                state['status'] = 'failed'
                state['failure_reason'] = failures[0].get(
                    'failure_reason', 'queue replication failed'
                )
                active.remove(loc_str)
                continue

            final_details, _ = _aggregate_simulation_samples(samples)
            state['final_details'] = final_details
            gap, selected = _relative_gap(final_details)
            state['history'].append(gap)
            if gap <= float(alpha) or selected is None:
                state['converged'] = gap <= float(alpha)
                active.remove(loc_str)
                continue

            group_key, routes, _maximum_used, _minimum = selected
            od, vehicle_type = group_key
            counts = state['assignments_ch'] if vehicle_type == 'F2' else state['assignments_no']
            current = counts[od]
            used_indices = [
                index for index, route in enumerate(routes)
                if route.get('used') and current[index] > 0
            ]
            if not routes or not used_indices:
                active.remove(loc_str)
                continue
            min_index = min(range(len(routes)), key=lambda index: routes[index]['travel_time'])
            max_index = max(used_indices, key=lambda index: routes[index]['travel_time'])
            if min_index == max_index:
                active.remove(loc_str)
                continue
            current[max_index] -= 1
            current[min_index] += 1

    assignments = {}
    convergence_data = {}
    for loc_str, state in states.items():
        iterations_completed = len(state['iteration_timings'])
        if (
            state['status'] == 'ok'
            and not state['converged']
            and iterations_completed >= int(max_ne_iterations)
        ):
            # A bounded Nash search that exhausts its iteration budget is a
            # completed simulation job, but it is not an equilibrium.  Keep
            # the artifact for diagnosis/comparison and expose this status so
            # downstream reports cannot mistake it for a converged result.
            state['status'] = 'nonconverged'
            state['failure_reason'] = (
                f'relative Nash gap {state["history"][-1] if state["history"] else float("inf"):.6g} '
                f'exceeded alpha={float(alpha):.6g} after {iterations_completed} iterations'
            )
        result = {
            'assignments': {
                'F1': state['assignments_no'],
                'F2': state['assignments_ch'],
            },
            'converged': state['converged'],
            'iterations': iterations_completed,
            'final_gap': state['history'][-1] if state['history'] else float('inf'),
            'route_metrics': state['final_details'],
            'flow_data': state['flow_data'],
            'iteration_timings': state['iteration_timings'],
            'status': state['status'],
            'failure_reason': state['failure_reason'],
            'resumed_complete': bool(state.get('resumed_complete', False)),
            'route_assignments': [
                {
                    'od_id': f'{od[0]}_{od[1]}',
                    'vehicle_type': vehicle_type,
                    'charger_id': entry.get('station node') if vehicle_type == 'F2' else None,
                    'route_id': entry.get(
                        'route_id', f'{od[0]}_{od[1]}_{vehicle_type}_{index}'
                    ),
                    'route_index': index,
                    'count': int(count),
                }
                for od, group in state['flow_data'].items()
                for vehicle_type, entries, counts in (
                    ('F1', group['no charging type'], state['assignments_no'].get(od, [])),
                    ('F2', group['charging type'], state['assignments_ch'].get(od, [])),
                )
                for index, (entry, count) in enumerate(zip(entries, counts))
            ],
            'network_hash': manifest['network_hash'],
        }
        assignments[loc_str] = result
        convergence_data[loc_str] = state['history']

    ne_path = os.path.join(work_dir, 'NE_path_assignments.pkl')
    with open(ne_path, 'wb') as handle:
        pickle.dump(assignments, handle)
    manifest = {
        'network_hash': manifest['network_hash'],
        'configuration_count': len(configs),
        'successful_configurations': sum(
            result.get('status', 'ok') != 'failed' for result in assignments.values()
        ),
        'failed_configurations': {
            key: value.get('failure_reason')
            for key, value in assignments.items()
            if value.get('status') == 'failed'
        },
        'nonconverged_configurations': {
            key: value.get('failure_reason')
            for key, value in assignments.items()
            if value.get('status') == 'nonconverged'
        },
        'iteration_timings': {
            key: value.get('iteration_timings', [])
            for key, value in assignments.items()
        },
        'alpha': alpha,
        'max_iterations': max_ne_iterations,
        'workers': pool_size,
        'workers_requested': workers,
        'workers_available': available_workers,
        'parallel_granularity': 'configuration_iteration_replication',
        'replications_per_iteration': int(num_iters),
        'resumed_configurations': sorted(
            loc_str for loc_str, state in states.items()
            if state.get('resumed_complete')
        ),
    }
    with open(os.path.join(work_dir, 'queue_manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2)
    if manifest['failed_configurations']:
        first_config, reason = next(iter(manifest['failed_configurations'].items()))
        raise RuntimeError(
            f'Queue Nash failed for charger configuration {first_config}: {reason}'
        )
    print(f'\nNash assignments saved to {ne_path}')
    return ne_path, convergence_data
