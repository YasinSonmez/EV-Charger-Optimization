#!/usr/bin/env python3
"""Measure simple one-run-per-flow-level BPR throughput by worker count."""

from __future__ import annotations

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from queue_sim.bpr_data_generator import generate_bpr_data
from src.config import Config
from src.network_pruning import ROAD_PROFILES
from src.road_network import RoadNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--workers', default='1,2,4,8,16')
    parser.add_argument('--max-links', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()

    config = Config.from_json(args.config)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    road_filter = config.road_filter
    highway_types = road_filter.get('highway_types') or list(
        ROAD_PROFILES[road_filter['road_profile']]
    )
    road_net = RoadNet('bpr-worker-benchmark')
    road_net.get_map(
        *config.coordinates, highway_types=highway_types,
        merge_chains=road_filter.get('merge_chains', True),
        prune_dead_ends=road_filter.get('prune_dead_ends', False),
        intersection_tolerance=road_filter.get('intersection_tolerance', 5),
        cache_policy=config.network.get('cache_policy', 'reuse'),
    )
    artifact = root / 'network'
    road_net.export_artifact(artifact, source={'benchmark_config': args.config})
    links = road_net.edges.sort_values(
        ['lanes', 'link_id'], ascending=[False, True], kind='mergesort'
    )
    active = links['link_id'].astype(int).head(args.max_links).tolist()
    bpr = config.pipeline['bpr_generation']
    rows = []
    baseline = None
    for workers in [int(value) for value in args.workers.split(',') if value.strip()]:
        work_dir = root / f'workers_{workers:02d}'
        if work_dir.exists() and not args.resume:
            raise FileExistsError(
                f'{work_dir} already exists; use a new output directory or --resume'
            )
        work_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        status, reason = 'complete', ''
        try:
            generate_bpr_data(
                coordinates=config.coordinates, work_dir=str(work_dir),
                road_net=road_net, artifact_dir=str(artifact), workers=workers,
                failure_policy='fail_fast', allow_proxy=False,
                seed=config.pipeline.get('random_seed', 42),
                mode='capacity_fraction_strict',
                flow_fractions=bpr['flow_fractions'],
                capacity_source=bpr.get('capacity_source', 'simulator'),
                capacity_per_lane=bpr.get('capacity_per_lane', 1900.0),
                calibration_window_hours=bpr.get('calibration_window_hours', 0.1),
                route_mode=bpr.get('route_mode', 'link_probe'),
                simulation_horizon=bpr.get('simulation_horizon', 10801),
                active_link_ids=active, resume=args.resume,
            )
        except Exception as exc:
            status, reason = 'failed', str(exc)
        elapsed = time.perf_counter() - started
        if baseline is None and status == 'complete':
            baseline = elapsed
        size = sum(path.stat().st_size for path in work_dir.rglob('*') if path.is_file())
        rows.append({
            'workers': workers, 'links': len(active), 'flow_levels': len(bpr['flow_fractions']),
            'status': status, 'wall_seconds': elapsed,
            'speedup': baseline / elapsed if baseline else '',
            'efficiency': baseline / elapsed / workers if baseline else '',
            'peak_rss_raw': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'output_bytes': size, 'failure_reason': reason,
        })
    with (root / 'worker_scaling.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / 'benchmark_manifest.json').write_text(json.dumps({
        'config': str(Path(args.config).resolve()), 'active_link_ids': active,
        'results': rows,
        'memory_note': 'peak_rss_raw is the parent-process high-water mark and excludes aggregate child RSS',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
