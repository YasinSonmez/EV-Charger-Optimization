#!/usr/bin/env python3
"""Calibrate simple BPR sample count and legacy Nash replication count."""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from queue_sim.find_nash import (
    _aggregate_simulation_samples,
    _relative_gap,
    _simulate_rep,
)
from src.model_fitter import model
from src.run_state import available_cpus


def parse_vector(value):
    return np.asarray(ast.literal_eval(value), dtype=float)


def fit_fixed_curve(x, y, capacity, fft):
    if np.ptp(y) <= 0.01 * max(abs(fft), 1e-9):
        return 0.0, 1.0

    def fixed(flow, alpha, beta):
        return model(flow, alpha, beta, capacity, fft)

    parameters, _ = curve_fit(
        fixed, x, y, p0=[0.15, 4.0],
        bounds=([0.0, 0.2], [100.0, 12.0]), maxfev=100000,
    )
    return tuple(map(float, parameters))


def bpr_study(source, sample_counts):
    frame = pd.read_csv(source)
    details = []
    per_link = []
    for row in frame.itertuples(index=False):
        x = parse_vector(row.x_vector)
        y = parse_vector(row.y_vector)
        capacity = float(row.calibration_capacity)
        fft = max(float(y[0]), 1e-9)
        try:
            full_parameters = fit_fixed_curve(x, y, capacity, fft)
            full_prediction = model(x, *full_parameters, capacity, fft)
        except (RuntimeError, ValueError, FloatingPointError):
            continue
        for count in sample_counts:
            indices = np.unique(np.rint(np.linspace(0, len(x) - 1, count)).astype(int))
            try:
                parameters = fit_fixed_curve(x[indices], y[indices], capacity, fft)
                prediction = model(x, *parameters, capacity, fft)
                scale = max(float(np.mean(np.abs(y))), 1e-9)
                observed_nrmse = float(np.sqrt(np.mean((prediction - y) ** 2)) / scale)
                reference_nrmse = float(
                    np.sqrt(np.mean((prediction - full_prediction) ** 2))
                    / max(float(np.mean(np.abs(full_prediction))), 1e-9)
                )
                high_flow_relative_error = float(
                    abs(prediction[-1] - full_prediction[-1])
                    / max(abs(float(full_prediction[-1])), 1e-9)
                )
                failed = False
            except (RuntimeError, ValueError, FloatingPointError):
                observed_nrmse = reference_nrmse = high_flow_relative_error = float('inf')
                failed = True
            per_link.append({
                'link_id': int(row.link_id), 'samples': count,
                'observed_nrmse': observed_nrmse,
                'reference_nrmse': reference_nrmse,
                'high_flow_relative_error': high_flow_relative_error,
                'failed': failed,
            })
    link_frame = pd.DataFrame(per_link)
    for count, group in link_frame.groupby('samples'):
        finite = group.loc[~group['failed']]
        details.append({
            'samples': int(count), 'links': int(len(group)),
            'fit_failures': int(group['failed'].sum()),
            'median_reference_nrmse': float(finite['reference_nrmse'].median()),
            'p95_reference_nrmse': float(finite['reference_nrmse'].quantile(0.95)),
            'max_reference_nrmse': float(finite['reference_nrmse'].max()),
            'median_observed_nrmse': float(finite['observed_nrmse'].median()),
            'p95_observed_nrmse': float(finite['observed_nrmse'].quantile(0.95)),
            'p95_high_flow_relative_error': float(
                finite['high_flow_relative_error'].quantile(0.95)
            ),
        })
    return pd.DataFrame(details).sort_values('samples'), link_frame


def action_from_details(details):
    gap, selected = _relative_gap(details)
    if selected is None:
        return float(gap), None
    group_key, routes, _maximum, _minimum = selected
    used = [index for index, route in enumerate(routes) if route.get('used')]
    if not used:
        return float(gap), None
    source = max(used, key=lambda index: routes[index]['travel_time'])
    target = min(range(len(routes)), key=lambda index: routes[index]['travel_time'])
    return float(gap), (group_key, source, target)


def action_label(action):
    return None if action is None else f'{action[0]}|{action[1]}->{action[2]}'


def action_value(action, reference_details):
    if action is None:
        return 0.0
    group_key, source, target = action
    routes = reference_details.get(group_key, [])
    if max(source, target) >= len(routes):
        return float('-inf')
    source_time = float(routes[source]['travel_time'])
    target_time = float(routes[target]['travel_time'])
    return max(0.0, (source_time - target_time) / max(target_time, 1e-9))


def route_vector(details):
    return np.asarray([
        float(route['travel_time'])
        for key in sorted(details, key=str)
        for route in details[key]
    ])


def queue_study(run_dir, configurations, sample_counts, bootstrap_draws, workers, output):
    import pickle

    run_dir = Path(run_dir)
    with (run_dir / 'queue' / 'NE_path_assignments.pkl').open('rb') as handle:
        assignments = pickle.load(handle)
    input_paths = tuple(str(run_dir / 'queue' / name) for name in (
        'canonical_nodes.csv', 'canonical_edges.csv', 'canonical_od.csv'
    ))
    maximum = max(sample_counts)
    jobs = []
    for name in configurations:
        state = assignments[name]
        charger_locs = tuple(int(value) for value in name.split(','))
        assigned = state['assignments']
        for rep in range(maximum):
            jobs.append((
                charger_locs, state['flow_data'], assigned['F2'], assigned['F1'],
                input_paths, str(output / 'queue_raw' / name),
                250, 250, 250, 0, 10801, 42, f'calibration_{name}', rep,
            ))
    started = time.perf_counter()
    with Pool(min(workers, len(jobs))) as pool:
        samples = list(pool.imap_unordered(_simulate_rep, jobs, chunksize=1))
    wall_seconds = time.perf_counter() - started
    failed = [sample for sample in samples if sample['status'] != 'ok']
    if failed:
        raise RuntimeError(f'queue calibration failed: {failed[0]}')
    grouped = {name: [] for name in configurations}
    for sample in samples:
        name = sample['scenario_name'].removeprefix('calibration_')
        grouped[name].append(sample)
    rng = np.random.default_rng(20260904)
    rows = []
    for name, values in grouped.items():
        values.sort(key=lambda item: item['rep'])
        reference_details, _ = _aggregate_simulation_samples(values)
        reference_gap, reference_action = action_from_details(reference_details)
        reference_routes = route_vector(reference_details)
        for count in sample_counts:
            records = []
            for _ in range(bootstrap_draws):
                indices = rng.integers(0, len(values), size=count)
                selected = [values[int(index)] for index in indices]
                details, _ = _aggregate_simulation_samples(selected)
                gap, action = action_from_details(details)
                routes = route_vector(details)
                records.append((
                    gap, action,
                    float(np.mean(np.abs(routes - reference_routes)
                                  / np.maximum(np.abs(reference_routes), 1e-9))),
                    max(0.0, reference_gap - action_value(action, reference_details)),
                ))
            gap_errors = np.asarray([abs(item[0] - reference_gap) for item in records])
            route_errors = np.asarray([item[2] for item in records])
            rows.append({
                'configuration': name, 'replications': count,
                'reference_gap': reference_gap,
                'reference_action': action_label(reference_action),
                'action_agreement': float(np.mean([
                    item[1] == reference_action for item in records
                ])),
                'median_gap_absolute_error': float(np.median(gap_errors)),
                'p95_gap_absolute_error': float(np.quantile(gap_errors, 0.95)),
                'median_route_mape': float(np.median(route_errors)),
                'p95_route_mape': float(np.quantile(route_errors, 0.95)),
                'p95_reference_action_regret': float(np.quantile(
                    [item[3] for item in records], 0.95
                )),
            })
    return pd.DataFrame(rows), wall_seconds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=None)
    parser.add_argument('--bootstrap-draws', type=int, default=500)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bpr_counts = [5, 7, 9, 13, 17, 25]
    queue_counts = [1, 2, 5, 10, 20, 50, 100]
    bpr_summary, bpr_links = bpr_study(
        'results/2026-07-25_19-44-24_n=3_chargers=2/'
        'bpr_dense_0_1_full95/bpr_data.csv', bpr_counts,
    )
    queue_summary, queue_wall = queue_study(
        'results/2026-07-25_19-44-24_n=3_chargers=2',
        ['0', '0,2', '1,2'], queue_counts, args.bootstrap_draws,
        args.workers or available_cpus(), args.output,
    )
    bpr_summary.to_csv(args.output / 'bpr_sample_summary.csv', index=False)
    bpr_links.to_csv(args.output / 'bpr_per_link.csv.gz', index=False, compression='gzip')
    queue_summary.to_csv(args.output / 'queue_replication_summary.csv', index=False)
    aggregate = queue_summary.groupby('replications').agg(
        minimum_action_agreement=('action_agreement', 'min'),
        maximum_p95_gap_error=('p95_gap_absolute_error', 'max'),
        maximum_p95_route_mape=('p95_route_mape', 'max'),
        maximum_p95_action_regret=('p95_reference_action_regret', 'max'),
    ).reset_index()
    aggregate.to_csv(args.output / 'queue_replication_aggregate.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(bpr_summary['samples'], bpr_summary['p95_reference_nrmse'], marker='o')
    axes[0].set(xlabel='BPR flow levels', ylabel='p95 curve deviation', title='BPR sample stability')
    axes[1].plot(aggregate['replications'], aggregate['minimum_action_agreement'], marker='o')
    axes[1].axhline(0.95, color='red', linestyle='--', linewidth=1)
    axes[1].set(xlabel='Queue replications / iteration', ylabel='Worst action agreement',
                title='Better-response decision stability', ylim=(0, 1.02))
    fig.tight_layout()
    fig.savefig(args.output / 'sample_count_calibration.png', dpi=180)
    plt.close(fig)
    (args.output / 'calibration_manifest.json').write_text(json.dumps({
        'bpr_source': 'saved 95-link, 104-level dense simulation',
        'queue_source': 'saved 33-node/95-link final assignments',
        'queue_configurations': ['0', '0,2', '1,2'],
        'queue_simulations': 3 * max(queue_counts),
        'queue_wall_seconds': queue_wall,
        'bootstrap_draws': args.bootstrap_draws,
        'workers': args.workers or available_cpus(),
    }, indent=2) + '\n')
    print('BPR\n', bpr_summary.to_string(index=False))
    print('QUEUE\n', aggregate.to_string(index=False))
    print(f'Artifacts: {args.output}')


if __name__ == '__main__':
    main()
