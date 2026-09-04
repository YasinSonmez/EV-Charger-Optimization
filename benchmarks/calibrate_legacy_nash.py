#!/usr/bin/env python3
"""Run bounded corrected better-response trajectories at several replication counts."""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import shutil
import sys
import tempfile
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from queue_sim.find_nash import (  # noqa: E402
    _aggregate_simulation_samples, _assignment_signature,
    _relative_gap, _simulate_rep,
)
from src.run_state import available_cpus  # noqa: E402


def run_trajectory(name, state, replications, max_iterations, workers, input_paths, raw_root):
    chargers = tuple(map(int, name.split(',')))
    assignments_no = copy.deepcopy(state['assignments']['F1'])
    assignments_ch = copy.deepcopy(state['assignments']['F2'])
    seen = set()
    history = []
    started = time.perf_counter()
    status = 'iteration_cap_reached'
    for iteration in range(max_iterations):
        signature = _assignment_signature(assignments_no, assignments_ch)
        if signature in seen:
            status = 'cycle_detected'
            break
        seen.add(signature)
        jobs = [(
            chargers, state['flow_data'], assignments_ch, assignments_no,
            input_paths, str(raw_root / name / f'iterations_{replications}_{iteration}'),
            250, 250, 250, 0, 10801, 42,
            f'legacy_{name}_{replications}_{iteration}', rep,
        ) for rep in range(replications)]
        with Pool(min(workers, replications)) as pool:
            samples = list(pool.imap_unordered(_simulate_rep, jobs, chunksize=1))
        failures = [sample for sample in samples if sample['status'] != 'ok']
        if failures:
            status = 'failed'
            break
        details, _ = _aggregate_simulation_samples(samples)
        gap, selected = _relative_gap(details)
        history.append(float(gap))
        if gap <= 0.01:
            status = 'converged'
            break
        if selected is None:
            status = 'no_move'
            break
        group_key, routes, _maximum, _minimum = selected
        od, vehicle_type = group_key
        counts = assignments_ch if vehicle_type == 'F2' else assignments_no
        current = counts[od]
        used = [index for index, route in enumerate(routes)
                if route.get('used') and current[index] > 0]
        source = max(used, key=lambda index: routes[index]['travel_time'])
        target = min(range(len(routes)), key=lambda index: routes[index]['travel_time'])
        if source == target:
            status = 'no_move'
            break
        current[source] -= 1
        current[target] += 1
    return {
        'configuration': name, 'replications': replications,
        'status': status, 'iterations': len(history),
        'final_gap': history[-1] if history else None,
        'minimum_gap': min(history) if history else None,
        'wall_seconds': time.perf_counter() - started,
        'history': history,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--counts', default='5,10,20,50')
    parser.add_argument('--configs', default='0,2;1,2')
    parser.add_argument('--max-iterations', type=int, default=100)
    parser.add_argument('--workers', type=int, default=None)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    run_dir = Path('results/2026-07-25_19-44-24_n=3_chargers=2')
    with (run_dir / 'queue' / 'NE_path_assignments.pkl').open('rb') as handle:
        saved = pickle.load(handle)
    input_paths = tuple(str(run_dir / 'queue' / name) for name in (
        'canonical_nodes.csv', 'canonical_edges.csv', 'canonical_od.csv'
    ))
    counts = [int(value) for value in args.counts.split(',')]
    configs = args.configs.split(';')
    workers = args.workers or available_cpus()
    temporary = Path(tempfile.mkdtemp(prefix='evopt-nash-calibration-'))
    try:
        rows = [
            run_trajectory(name, saved[name], count, args.max_iterations,
                           workers, input_paths, temporary)
            for name in configs for count in counts
        ]
    finally:
        shutil.rmtree(temporary)
    (args.output / 'nash_calibration.json').write_text(json.dumps(rows, indent=2) + '\n')
    import pandas as pd
    pd.DataFrame([{key: value for key, value in row.items() if key != 'history'}
                  for row in rows]).to_csv(args.output / 'nash_calibration.csv', index=False)
    print(pd.DataFrame([{key: value for key, value in row.items() if key != 'history'}
                        for row in rows]).to_string(index=False))
    print(f'Artifacts: {args.output}')


if __name__ == '__main__':
    main()
