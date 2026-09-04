#!/usr/bin/env python3
"""Unified EV Charger Optimization Pipeline.

Runs the complete end-to-end experiment:
  1. BPR fitting (TrafficModelFitter)
  2. Congestion-game equilibrium (outer_optimization)
  3. Queue NE assignments (find_nash_assignments)
  4. Queue comparison: greedy vs exhaustive (run_comparison)
  5. Report generation

Usage:
  python pipeline.py --config config.json
"""
import argparse
import io
import json
import os
import subprocess
import sys
import time
import pickle
import platform
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config, NetworkConfig
from src.contracts import SeedManager, TimingRecorder, stable_json
from src.network_artifact import load_network_artifact
from src.run_state import (
    atomic_write_json, available_cpus, config_digest,
    directory_inventory, process_provenance, safe_name,
)
from src.sanity_checks import validate_experiment_outputs
from src.model_fitter import TrafficModelFitter, convert_string_to_array, validate_bpr_fit_table
from src.utils import outer_optimization

try:
    from queue_sim import QUEUE_SIM_AVAILABLE, _QUEUE_SIM_ERROR
except (ImportError, ModuleNotFoundError):
    QUEUE_SIM_AVAILABLE = False
    _QUEUE_SIM_ERROR = "queue_sim package not found (macOS-only)"


def _fill_missing_links_to_count(pandas_df, target_count, allow_missing=False):
    """Validate BPR coverage; synthetic missing links require an explicit opt-in."""
    import pandas as pd
    import numpy as np
    existing = set(pandas_df['link_id'].unique())
    expected = set(range(target_count))
    missing = sorted(expected - existing)
    extra = sorted(existing - expected)
    if extra:
        raise ValueError(
            f"BPR data contains link_ids outside the canonical artifact: {extra[:10]}"
            + (" ..." if len(extra) > 10 else "")
        )
    if not missing:
        return pandas_df
    if not allow_missing:
        raise ValueError(
            f"BPR data is missing link_ids {missing[:10]}"
            + (" ..." if len(missing) > 10 else "")
            + ". Regenerate BPR data for the canonical network or explicitly enable degraded mode."
        )
    n_samples = len(pandas_df.iloc[0]['x_vector']) if 'x_vector' in pandas_df.columns else 25
    rows = []
    for lid in missing:
        rows.append({
            'link_id': lid,
            'x_vector': np.zeros(n_samples),
            'y_vector': np.zeros(n_samples),
            'a_fit': 0.0, 'b_fit': 0.0, 'cap_fit': 1.0, 'fft_fit': 1.0, 'R^2': np.nan,
            'fit_status': 'degraded_missing',
        })
    df = pd.concat([pandas_df, pd.DataFrame(rows)], ignore_index=True).sort_values('link_id').reset_index(drop=True)
    print(f"Filled {len(missing)} missing link_ids (up to {target_count})")
    return df


def _ensure_bpr_reference_metadata(pandas_df, artifact_dir, capacity_source='simulator', capacity_per_lane=1900.0):
    """Attach canonical FFT/capacity references when fitting old observation files."""
    if artifact_dir is None:
        return pandas_df
    nodes, edges, _ = load_network_artifact(artifact_dir)
    by_id = edges.set_index('link_id')
    output = pandas_df.copy()
    if 'calibration_fft' not in output.columns:
        output['calibration_fft'] = output['link_id'].map(
            lambda link_id: float(by_id.loc[int(link_id), 'length'])
            / (float(by_id.loc[int(link_id), 'maxmph']) / 2.2369)
        )
    if 'calibration_capacity' not in output.columns:
        if capacity_source == 'artifact':
            output['calibration_capacity'] = output['link_id'].map(
                lambda link_id: float(by_id.loc[int(link_id), 'capacity'])
            )
        else:
            output['calibration_capacity'] = output['link_id'].map(
                lambda link_id: float(by_id.loc[int(link_id), 'lanes']) * float(capacity_per_lane)
            )
    return output


def _ensure_bpr_link_length(pandas_df, artifact_dir):
    """Attach canonical link lengths required by CG by stable ``link_id``."""
    if artifact_dir is None:
        return pandas_df
    _, edges, _ = load_network_artifact(artifact_dir)
    by_id = edges.set_index(edges['link_id'].astype(int))
    output = pandas_df.copy()
    ids = output['link_id'].astype(int)
    missing = sorted(set(ids) - set(by_id.index.astype(int)))
    if missing:
        raise ValueError(
            'BPR table contains link_ids absent from canonical artifact while '
            f'attaching link_length: {missing[:10]}'
        )
    canonical_lengths = ids.map(by_id['length'].astype(float))
    if 'link_length' not in output.columns:
        output['link_length'] = canonical_lengths.to_numpy()
    else:
        existing = pd.to_numeric(output['link_length'], errors='coerce')
        output['link_length'] = existing.fillna(canonical_lengths).to_numpy()
    if not np.isfinite(output['link_length'].astype(float)).all():
        raise ValueError('BPR table contains non-finite canonical link lengths')
    return output


def _bpr_manifest_is_compatible(manifest_path, network_hash, bpr_config, seed):
    """Return whether a cached BPR artifact matches the current request."""
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return False
    requested_mode = bpr_config.get('mode', 'historical_artifact_compatible')
    checks = {
        'network_hash': network_hash,
        'bpr_mode': requested_mode,
        'num_samples': int(bpr_config.get('num_samples', 25)),
        'max_flow': float(bpr_config.get('max_flow', 250)),
        'random_seed': int(seed if seed is not None else 0),
        'fitter_version': (
            'historical_v1'
            if requested_mode == 'historical_artifact_compatible'
            else 'capacity_fraction_v1'
        ),
        'route_semantics': (
            'measured_target_flow_with_straight_ahead_context'
            if requested_mode == 'historical_artifact_compatible'
            else bpr_config.get('route_mode', 'link_probe')
        ),
        'fit_screening': bpr_config.get('fit_screening', 'none'),
        'correlation_threshold': float(bpr_config.get('correlation_threshold', 0.0)),
        'variation_ratio_threshold': float(bpr_config.get('variation_ratio_threshold', 0.0)),
        'accept_low_r2': bool(bpr_config.get('accept_low_r2', True)),
        'missing_context_policy': bpr_config.get('missing_context_policy', 'synthetic_boundary'),
        'synthetic_context_capacity_multiplier': float(
            bpr_config.get('synthetic_context_capacity_multiplier', 10.0)
        ),
        'synthetic_context_length_m': float(
            bpr_config.get('synthetic_context_length_m', 1.0)
        ),
        'simulation_horizon': int(bpr_config.get('simulation_horizon', 10801)),
    }
    for key, expected in checks.items():
        actual = manifest.get(key)
        if key in {
            'max_flow', 'correlation_threshold', 'variation_ratio_threshold',
            'synthetic_context_capacity_multiplier', 'synthetic_context_length_m',
        } and actual is not None:
            if not np.isclose(float(actual), expected):
                return False
        elif actual != expected:
            return False
    return True


def _validate_bpr_network_rows(pandas_df, network_hash):
    """Require row-level provenance when BPR is attached to an artifact."""
    if network_hash is None:
        return
    if 'network_hash' not in pandas_df.columns:
        raise ValueError('BPR table is missing required network_hash provenance')
    values = set(pandas_df['network_hash'].dropna().astype(str))
    if values != {str(network_hash)}:
        raise ValueError(
            f'BPR row network_hash mismatch: expected {network_hash}, got {sorted(values)}'
        )


def load_or_fit_model(data_path="data/traffic_data.csv", cache_path="data/cached_results.pkl",
                      coordinates=None, bpr_config=None, work_dir=None, n_links=None,
                      road_filter_config=None, road_net=None, artifact_dir=None,
                      seed_manager=None, allow_generate=True):
    """Load cached BPR fit, fit from existing data, or generate data + fit.

    Priority:
    1. If cache exists → load it
    2. If traffic_data.csv exists → fit from it
    3. If coordinates provided → generate data via queue sim, then fit

    Args:
        n_links: If provided, ensure the fit covers link_ids 0..n_links-1.
        road_filter_config: Dict with 'highway_types' and 'prune_dead_ends' keys.
    """
    bpr_config = dict(bpr_config or {})
    bpr_mode = bpr_config.get('mode', 'historical_artifact_compatible')
    validation_mode = bpr_config.get(
        'fit_validation',
        'parameter_complete' if bpr_mode == 'historical_artifact_compatible' else 'full',
    )
    force_regenerate = bool(bpr_config.get('force_regenerate', False)) and allow_generate
    require_full_fit = bool(bpr_config.get('require_full_fit', False)) or validation_mode == 'full'
    r2_threshold = float(bpr_config.get('min_r2', 0.5))
    fixed_references = bool(bpr_config.get('fixed_references', False))
    expected_link_ids = None
    if road_net is not None and hasattr(road_net, 'edges'):
        expected_link_ids = road_net.edges['link_id'].astype(int).tolist()

    manifest_path = os.path.join(work_dir or os.path.dirname(cache_path) or '.', 'bpr_manifest.json')
    network_hash = None
    if artifact_dir is not None:
        _, _, artifact_manifest = load_network_artifact(artifact_dir)
        network_hash = artifact_manifest['network_hash']
    cache_compatible = (
        artifact_dir is None
        or _bpr_manifest_is_compatible(
            manifest_path, network_hash, bpr_config,
            seed_manager.seed if seed_manager is not None else 0,
        )
    )

    if os.path.exists(cache_path) and not force_regenerate and cache_compatible:
        with open(cache_path, "rb") as f:
            pandas_df, model_fitter = pickle.load(f)
        _validate_bpr_network_rows(pandas_df, network_hash)
        if n_links:
            pandas_df = _fill_missing_links_to_count(
                pandas_df, n_links,
                allow_missing=(bpr_config or {}).get('allow_missing_links', False),
            )
        validate_bpr_fit_table(
            pandas_df,
            expected_link_ids=expected_link_ids,
            require_full_fit=require_full_fit,
            validation_mode=validation_mode,
        )
        print("Loaded cached BPR fit results.")
    elif os.path.exists(data_path) and not force_regenerate and cache_compatible:
        print("No cache found. Fitting BPR models from existing data...")
        import pandas as pd
        pandas_df = pd.read_csv(data_path)
        convert_string_to_array(pandas_df, 'x_vector')
        convert_string_to_array(pandas_df, 'y_vector')
        _validate_bpr_network_rows(pandas_df, network_hash)
        if bpr_mode == 'capacity_fraction_strict':
            pandas_df = _ensure_bpr_reference_metadata(
                pandas_df, artifact_dir,
                capacity_source=bpr_config.get('capacity_source', 'simulator'),
                capacity_per_lane=bpr_config.get('capacity_per_lane', 1900.0),
            )
        model_fitter = TrafficModelFitter(pandas_df=pandas_df)
        model_fitter.parallel_fit_and_evaluate(
            workers=(bpr_config or {}).get('fit_workers'),
            output_dir=work_dir,
            save_plots=(bpr_config or {}).get('save_fit_plots', True),
            require_full_fit=require_full_fit,
            r2_threshold=r2_threshold,
            expected_link_ids=expected_link_ids,
            fixed_references=fixed_references,
            fit_mode=bpr_mode,
            validation_mode=validation_mode,
            fit_screening=bpr_config.get('fit_screening', 'none'),
            correlation_threshold=bpr_config.get('correlation_threshold', 0.0),
            variation_ratio_threshold=bpr_config.get('variation_ratio_threshold', 0.0),
            accept_low_r2=bpr_config.get('accept_low_r2', True),
        )
        model_fitter.fill_missing_link_ids()
        pandas_df = model_fitter.df
        if n_links:
            pandas_df = _fill_missing_links_to_count(
                pandas_df, n_links,
            allow_missing=bpr_config.get('allow_missing_links', False),
            )
        with open(cache_path, "wb") as f:
            pickle.dump((pandas_df, model_fitter), f)
        print("Cached BPR fit results.")
    elif allow_generate and coordinates is not None and (
        QUEUE_SIM_AVAILABLE
        or (bpr_mode == 'historical_artifact_compatible'
            and (bpr_config.get('failure_policy') == 'proxy'
                 or bpr_config.get('allow_proxy', False)))
    ):
        print("No cache or data found. Generating BPR data via queue simulation...")
        from queue_sim.bpr_data_generator import generate_and_save_bpr_data
        num_samples = bpr_config.get('num_samples', 25)
        max_flow = bpr_config.get('max_flow', 250)
        rf = road_filter_config or {}
        highway_types = rf.get('highway_types') if rf.get('enabled', True) else None
        _, n_links_generated = generate_and_save_bpr_data(
            coordinates, data_path,
            num_samples=num_samples, max_flow=max_flow, work_dir=work_dir,
            highway_types=highway_types,
            road_net=road_net,
            artifact_dir=artifact_dir,
            workers=bpr_config.get('workers'),
            failure_policy=bpr_config.get('failure_policy', 'fail_fast'),
            allow_proxy=bpr_config.get('allow_proxy', False),
            seed=(seed_manager.seed if seed_manager else 0),
            timeout=bpr_config.get('timeout'),
            flow_fractions=bpr_config.get('flow_fractions'),
            capacity_source=bpr_config.get('capacity_source', 'simulator'),
            capacity_per_lane=bpr_config.get('capacity_per_lane', 1900.0),
            calibration_window_hours=bpr_config.get('calibration_window_hours', 1.0),
            route_mode=bpr_config.get('route_mode', 'link_probe'),
            mode=bpr_mode,
            missing_context_policy=bpr_config.get('missing_context_policy', 'synthetic_boundary'),
            synthetic_context_capacity_multiplier=bpr_config.get(
                'synthetic_context_capacity_multiplier', 10.0
            ),
            synthetic_context_length_m=bpr_config.get(
                'synthetic_context_length_m', 1.0
            ),
            simulation_horizon=bpr_config.get(
                'simulation_horizon',
                10801,
            ),
            active_link_ids=bpr_config.get('active_link_ids'),
            resume=bpr_config.get('resume', True),
        )
        import pandas as pd
        pandas_df = pd.read_csv(data_path)
        convert_string_to_array(pandas_df, 'x_vector')
        convert_string_to_array(pandas_df, 'y_vector')
        _validate_bpr_network_rows(pandas_df, network_hash)
        if bpr_mode == 'capacity_fraction_strict':
            pandas_df = _ensure_bpr_reference_metadata(
                pandas_df, artifact_dir,
                capacity_source=bpr_config.get('capacity_source', 'simulator'),
                capacity_per_lane=bpr_config.get('capacity_per_lane', 1900.0),
            )
        model_fitter = TrafficModelFitter(pandas_df=pandas_df)
        model_fitter.parallel_fit_and_evaluate(
            workers=(bpr_config or {}).get('fit_workers'),
            output_dir=work_dir,
            save_plots=(bpr_config or {}).get('save_fit_plots', True),
            require_full_fit=require_full_fit,
            r2_threshold=r2_threshold,
            expected_link_ids=expected_link_ids,
            fixed_references=fixed_references,
            fit_mode=bpr_mode,
            validation_mode=validation_mode,
            fit_screening=bpr_config.get('fit_screening', 'none'),
            correlation_threshold=bpr_config.get('correlation_threshold', 0.0),
            variation_ratio_threshold=bpr_config.get('variation_ratio_threshold', 0.0),
            accept_low_r2=bpr_config.get('accept_low_r2', True),
        )
        model_fitter.fill_missing_link_ids()
        pandas_df = model_fitter.df
        if n_links:
            pandas_df = _fill_missing_links_to_count(
                pandas_df, n_links,
            allow_missing=bpr_config.get('allow_missing_links', False),
            )
        elif n_links_generated:
            pandas_df = _fill_missing_links_to_count(
                pandas_df, n_links_generated,
            allow_missing=bpr_config.get('allow_missing_links', False),
            )
        with open(cache_path, "wb") as f:
            pickle.dump((pandas_df, model_fitter), f)
        print("Generated BPR data and cached fit results.")
    else:
        raise FileNotFoundError(
            f"No BPR data found at {data_path} or {cache_path}. "
            f"Provide coordinates and run on macOS to generate data via queue simulation."
        )
    if artifact_dir is not None:
        pandas_df = _ensure_bpr_link_length(pandas_df, artifact_dir)
        bpr_manifest_path = os.path.join(
            work_dir or os.path.dirname(cache_path) or '.', 'bpr_manifest.json'
        )
        _, _, network_manifest = load_network_artifact(artifact_dir)
        pandas_df = pandas_df.copy()
        if 'observation_source' not in pandas_df.columns:
            source_series = pandas_df.get(
                'fit_status', pd.Series('simulated_contextual', index=pandas_df.index)
            )
            pandas_df['observation_source'] = source_series.map(
                lambda value: 'proxy' if value == 'proxy' else 'simulated_contextual'
            )
        pandas_df['network_hash'] = network_manifest['network_hash']
        pandas_df['bpr_mode'] = bpr_mode
        pandas_df['sample_count'] = pandas_df['x_vector'].apply(
            lambda values: int(len(values)) if hasattr(values, '__len__') else 0
        )
        model_fitter.df = pandas_df
        # ``parallel_fit_and_evaluate`` writes its diagnostic table before
        # the artifact identity is known at this orchestration layer.  Save
        # once more so the persisted fit table is self-identifying too.
        model_fitter.save_results_to_csv(
            os.path.join(work_dir or os.path.dirname(cache_path) or '.', 'fitter_results.csv')
        )
        with open(cache_path, "wb") as f:
            pickle.dump((pandas_df, model_fitter), f)
        bpr_manifest = {}
        if os.path.exists(bpr_manifest_path):
            try:
                with open(bpr_manifest_path) as handle:
                    bpr_manifest = json.load(handle)
            except (OSError, ValueError):
                bpr_manifest = {}
        bpr_manifest.update({
            'network_hash': network_manifest['network_hash'],
            'bpr_mode': bpr_mode,
            'fit_validation': validation_mode,
            'num_samples': int(bpr_config.get('num_samples', 25)),
            'max_flow': float(bpr_config.get('max_flow', 250)),
            'random_seed': int(seed_manager.seed if seed_manager is not None else 0),
            'fitter_version': (
                'historical_v1' if bpr_mode == 'historical_artifact_compatible'
                else 'capacity_fraction_v1'
            ),
            'historical_reference_commit': bpr_config.get('historical_reference_commit') if bpr_mode == 'historical_artifact_compatible' else None,
            'route_semantics': 'measured_target_flow_with_straight_ahead_context' if bpr_mode == 'historical_artifact_compatible' else bpr_config.get('route_mode', 'link_probe'),
            'missing_context_policy': bpr_config.get('missing_context_policy', 'synthetic_boundary'),
            'synthetic_context_capacity_multiplier': float(
                bpr_config.get('synthetic_context_capacity_multiplier', 10.0)
            ),
            'synthetic_context_length_m': float(
                bpr_config.get('synthetic_context_length_m', 1.0)
            ),
            'simulation_horizon': int(
                bpr_config.get(
                    'simulation_horizon',
                    10801,
                )
            ),
            'fit_screening': bpr_config.get('fit_screening', 'none'),
            'correlation_threshold': float(bpr_config.get('correlation_threshold', 0.0)),
            'variation_ratio_threshold': float(bpr_config.get('variation_ratio_threshold', 0.0)),
            'accept_low_r2': bool(bpr_config.get('accept_low_r2', True)),
            'link_count': int(len(pandas_df)),
            'source_data': data_path,
            'source_cache': cache_path,
            'fit_status_counts': pandas_df.get('fit_status', pd.Series(dtype=str)).value_counts().to_dict(),
            'observation_source_counts': pandas_df.get(
                'observation_source', pd.Series(dtype=str)
            ).value_counts().to_dict(),
            'degraded_observation_links': [
                int(row['link_id']) for _, row in pandas_df.iterrows()
                if row.get('observation_source') == 'proxy'
                or row.get('fit_status') in {'full_relaxed', 'constant_fallback'}
            ],
            'fit_execution': getattr(model_fitter, 'fit_metadata', {}),
        })
        with open(bpr_manifest_path, 'w') as handle:
            json.dump(bpr_manifest, handle, indent=2, default=str)
    return pandas_df, model_fitter


def _plot_objective_comparison(cg_results, queue_results, output_path):
    """Plot CG vs Queue objectives for each charger placement as grouped bar chart.

    Normalizes each model's objectives to its own best (min) so both models
    are comparable on the same scale. Highlights the best placement for each.
    """
    if not cg_results or not queue_results:
        return

    cg_configs = cg_results.get('all_configs', [])
    q_results = queue_results.get('exhaustive_results', [])

    cg_labels = [str(c['chargers']) for c in cg_configs]
    cg_vals = [c['objective'] for c in cg_configs]
    q_labels = [str(r['positions']) for r in q_results]
    q_vals = [r['avg_travel_time'] for r in q_results if r['avg_travel_time'] != float('inf')]

    if not cg_vals or not q_vals:
        return

    cg_min = min(cg_vals)
    q_min = min(q_vals)
    cg_norm = [v / cg_min for v in cg_vals]
    q_norm = [v / q_min for v in q_vals if v != float('inf')]

    def _placement_sort_key(label):
        try:
            values = tuple(int(value.strip()) for value in label.strip('[]').split(',') if value.strip())
            return (len(values), values)
        except (TypeError, ValueError):
            return (999, (label,))

    all_labels = sorted(set(cg_labels) | set(q_labels), key=_placement_sort_key)
    cg_map = dict(zip(cg_labels, cg_norm))
    q_map = {str(r['positions']): r['avg_travel_time'] / q_min for r in q_results if r['avg_travel_time'] != float('inf')}

    # Do not encode an unavailable comparison as zero.  The CG stage also
    # evaluates intermediate one-charger placements, while the queue
    # comparison normally reports only target-size exhaustive placements.
    # Missing bars are rendered as gaps and called out in the legend/title.
    cg_bars = [cg_map.get(l, np.nan) for l in all_labels]
    q_bars = [q_map.get(l, np.nan) for l in all_labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: normalized comparison
    x = np.arange(len(all_labels))
    width = 0.35
    ax1.bar(x - width/2, cg_bars, width, label='Congestion Game', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, q_bars, width, label='Queue Simulation', color='coral', alpha=0.8)
    ax1.set_ylabel('Normalized objective (best = 1.0)')
    missing_queue = sorted(set(all_labels) - set(q_labels))
    title = 'CG vs Queue: Objective per Placement (Normalized)'
    if missing_queue:
        title += '\n(queue result unavailable for intermediate CG placements)'
    ax1.set_title(title, fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Right: raw values (twin axis for different scales)
    cg_raw = [cg_map.get(label, np.nan) * cg_min for label in all_labels]
    ax2.bar(x - width/2, cg_raw, width, label='CG objective', color='steelblue', alpha=0.8)
    ax2.set_ylabel('CG total delay', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, rotation=45, ha='right')
    ax2.set_title('Raw Objectives (different scales)')

    ax2b = ax2.twinx()
    q_raw = [next((r['avg_travel_time'] for r in q_results if str(r['positions']) == l and r['avg_travel_time'] != float('inf')), np.nan) for l in all_labels]
    ax2b.bar(x + width/2, q_raw, width, label='Queue avg TT', color='coral', alpha=0.8)
    ax2b.set_ylabel('Queue avg travel time', color='coral')
    ax2b.tick_params(axis='y', labelcolor='coral')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Objective comparison plot saved to {output_path}")


def _save_convergence_csv(convergence_data, path):
    """Save NE convergence data (per-config, per-iteration diff) to CSV."""
    import csv
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['config', 'iteration', 'diff'])
        for config_str, diffs in convergence_data.items():
            for i, d in enumerate(diffs):
                writer.writerow([config_str, i, d])


def _plot_pruning_phases(network_stages, output_path, node_count=None, edge_count=None,
                         stage_maps=None):
    """Plot nodes/edges per cleaning phase (bar chart) + map grid of each stage.

    When *stage_maps* is provided, the figure has two rows:
      - Top row: bar chart (nodes & edges per phase)
      - Bottom row: one panel per stage showing node positions + edge lines
    Without *stage_maps*, only the bar chart is shown.
    When *network_stages* is None (no cleaning), draws a single fallback bar.
    """
    has_maps = bool(stage_maps)
    if network_stages:
        phases = sorted(network_stages.keys())
        nodes = [network_stages[p].get('nodes', 0) for p in phases]
        edges = [network_stages[p].get('edges', 0) for p in phases]
    else:
        n = node_count or 0
        e = edge_count or 0
        phases = ['original (no cleaning)']
        nodes = [n]
        edges = [e]

    if has_maps:
        map_phases = sorted(stage_maps.keys())
        n_maps = len(map_phases)
        ncols = min(n_maps, 3)
        nrows = 2 + (n_maps + ncols - 1) // ncols
        fig = plt.figure(figsize=(4 * ncols, 5 + 4 * (nrows - 2)))
        ax_bar = fig.add_subplot(2, 1, 1)
    else:
        fig, ax_bar = plt.subplots(figsize=(10, 5))

    # Bar chart (always)
    x = np.arange(len(phases))
    w = 0.35
    ax_bar.bar(x - w/2, nodes, w, color='#1a9850', label='Nodes')
    ax_bar.bar(x + w/2, edges, w, color='#3288bd', label='Edges')
    ax_bar.set_ylabel('Count')
    ax_bar.set_title('Network Cleaning Pipeline — Nodes & Edges per Phase')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(phases, rotation=30, ha='right', fontsize=8)
    ax_bar.legend()
    for bar, val in zip(ax_bar.patches, nodes + edges):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(nodes)*0.01,
                    str(val), ha='center', va='bottom', fontsize=7)

    # Map grid (when available)
    if has_maps:
        for idx, phase in enumerate(map_phases):
            ax = fig.add_subplot(nrows, ncols, ncols * 2 + idx + 1)
            mp = stage_maps[phase]
            xs = [p[0] for p in mp['nodes_xy']]
            ys = [p[1] for p in mp['nodes_xy']]
            if not xs or not ys:
                ax.text(0.5, 0.5, 'empty', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(phase, fontsize=8)
                continue
            pos_by_id = {nid: mp['nodes_xy'][i] for i, nid in enumerate(mp['_node_ids'])}
            geoms = mp.get('edges_geom', [])
            if geoms and len(geoms) == len(mp['edges_pairs']):
                for coords in geoms:
                    if len(coords) >= 2:
                        ax.plot([c[0] for c in coords], [c[1] for c in coords],
                                linewidth=0.3, color='gray', alpha=0.5, zorder=1)
            else:
                for u, v in mp['edges_pairs']:
                    if u in pos_by_id and v in pos_by_id:
                        ax.plot([pos_by_id[u][0], pos_by_id[v][0]],
                                [pos_by_id[u][1], pos_by_id[v][1]],
                                linewidth=0.3, color='gray', alpha=0.5, zorder=1)
            ax.scatter(xs, ys, s=3, c='#1a9850', alpha=0.8, zorder=2)
            ax.set_title(f"{phase}\nnodes={mp['_n']} edges={mp['_e']}",
                         fontsize=7)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xticks([])
            ax.set_yticks([])

        for idx in range(n_maps, ncols * (nrows - 2)):
            fig.add_subplot(nrows, ncols, ncols * 2 + idx + 1).set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_bpr_fit_samples(pandas_df, output_path, n_random=20, n_worst=10, seed=0):
    """Plot sample BPR fit diagnostics: random links + worst-R² links."""
    if pandas_df is None or 'R^2' not in pandas_df.columns:
        return
    import random
    df = pandas_df.copy()
    idxs = list(range(len(df)))
    random.Random(seed).shuffle(idxs)
    random_sample = df.iloc[idxs[:n_random]]
    worst = df.nsmallest(n_worst, 'R^2')
    sample = pd.concat([random_sample, worst])
    sample = sample.loc[~sample.index.duplicated(keep='first')]
    n = min(len(sample), 30)

    rows = 5; cols = 6
    fig, axes = plt.subplots(rows, cols, figsize=(18, 14))
    axes = axes.flatten()
    for i in range(rows * cols):
        if i < n:
            row = sample.iloc[i]
            ax = axes[i]
            if 'x_vector' in row and 'y_vector' in row:
                try:
                    xv = row['x_vector'] if hasattr(row['x_vector'], '__len__') else []
                    yv = row['y_vector'] if hasattr(row['y_vector'], '__len__') else []
                    if len(xv) > 0:
                        ax.scatter(xv, yv, s=5, alpha=0.7)
                        a_fit = row.get('a_fit', np.nan)
                        fft_fit = row.get('fft_fit', np.nan)
                        if np.isfinite(a_fit) and np.isfinite(fft_fit):
                            if float(a_fit) > 0:
                                xs = np.linspace(min(xv), max(xv), 100)
                                a, b, c, f = a_fit, row['b_fit'], row['cap_fit'], fft_fit
                                ys = f * (1 + a * (xs/c)**b)
                                ax.plot(xs, ys, 'r-', linewidth=1, label='Full BPR fit')
                            else:
                                ax.axhline(
                                    y=float(fft_fit), color='darkorange',
                                    linewidth=1, linestyle='--',
                                    label='Constant fit',
                                )
                except Exception:
                    pass
            r2 = row.get('R^2', np.nan)
            status = row.get('fit_status', 'unknown')
            source = row.get('observation_source', 'unknown')
            ax.set_title(
                f'Link {int(row["link_id"])} {status}/{source} R²={r2:.3f}',
                fontsize=6,
            )
        ax.set_xticks([]); ax.set_yticks([])
    for i in range(n, rows*cols):
        axes[i].set_visible(False)
    fig.suptitle('BPR Fit Diagnostics — Random + Worst-R² Links', fontsize=12, fontweight='bold')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_historical_bpr_comparison(pandas_df, output_path,
                                    reference_commit='37eab33'):
    """Compare fresh historical-compatible diagnostics with the reference.

    The historical network is a regression reference only.  Link IDs are not
    compared across topologies; the plot compares sample counts, fit-status
    counts, and R² distributions instead.
    """
    if pandas_df is None or 'R^2' not in pandas_df.columns:
        return
    fresh = pd.to_numeric(pandas_df['R^2'], errors='coerce').dropna().to_numpy()
    reference = np.array([], dtype=float)
    reference_status = {}
    try:
        raw = subprocess.check_output(
            ['git', 'show', f'{reference_commit}:data/fitter_results.csv'],
            stderr=subprocess.DEVNULL,
        )
        old = pd.read_csv(io.BytesIO(raw))
        reference = pd.to_numeric(old['R^2'], errors='coerce').dropna().to_numpy()
        reference_status = {
            'full': int(((old['a_fit'] > 0) & (old['b_fit'] > 0)).sum()),
            'constant_fallback': int(((old['a_fit'] == 0) & (old['b_fit'] == 0)).sum()),
        }
    except (OSError, subprocess.CalledProcessError, ValueError, KeyError):
        pass

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    labels = ['fresh current network']
    axes[0].bar(labels, [len(fresh)], color='steelblue', label='fresh')
    if reference.size:
        axes[0].bar(['historical reference'], [len(reference)], color='darkorange', label='reference')
    axes[0].set_ylabel('links with fitted R²')
    axes[0].set_title('BPR coverage')
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].legend(fontsize=8)

    bins = np.linspace(0, 1.01, 21)
    if reference.size:
        axes[1].hist(reference, bins=bins, alpha=0.55, label=f'reference {reference_commit}')
    if fresh.size:
        axes[1].hist(fresh, bins=bins, alpha=0.55, label='fresh current network')
    axes[1].set_xlabel('R² (legacy-compatible field)')
    axes[1].set_ylabel('link count')
    axes[1].set_title('Historical-compatible fit quality')
    axes[1].legend(fontsize=8)
    fresh_status = pandas_df.get('fit_status', pd.Series(dtype=str)).value_counts().to_dict()
    fig.suptitle(
        'Historical BPR compatibility — status comparison\n'
        f'fresh={fresh_status}; reference={reference_status or "unavailable"}',
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_ne_convergence(convergence_data, output_path):
    """Plot NE convergence curves: diff vs iteration per config."""
    if not convergence_data:
        return
    fig, ax = plt.subplots(figsize=(10, 6))
    for config_str, diffs in convergence_data.items():
        if diffs:
            ax.plot(range(len(diffs)), diffs, linewidth=1, alpha=0.7, label=config_str[:20])
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative route travel-time gap')
    ax.set_title('NE Convergence — Better-Response Heuristic')
    if len(convergence_data) <= 15:
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_timing_breakdown(timing, output_path):
    """Plot timing breakdown as horizontal bar chart."""
    steps = {k: v for k, v in timing.items() if k != 'total' and v > 0}
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(steps.keys())
    values = list(steps.values())
    colors = ['#1a9850', '#66bd63', '#3288bd', '#542788', '#d73027'][:len(names)]
    bars = ax.barh(names, values, color=colors)
    total = sum(values)
    for bar, val in zip(bars, values):
        pct = val / total * 100 if total > 0 else 0
        ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}s ({pct:.0f}%)', va='center', fontsize=8)
    ax.set_xlabel('Wall-clock time (s)')
    ax.set_title(f'Timing Breakdown (total: {total:.0f}s)')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _generate_run_summary(experiment_dir, config, timing, cg_results, queue_results,
                          convergence_data, network_stages):
    """Write a unified plain-text run_summary.txt at the experiment root."""
    lines = [
        "RUN SUMMARY",
        "===========",
        f"Timestamp:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Platform:   {platform.system()}",
        f"Coords:     {config.coordinates}",
        f"Chargers:   {config.num_chargers}",
        f"Positions:  {config.possible_charger_positions}",
        "",
    ]

    lines.append("-" * 40)
    lines.append("NETWORK STAGES")
    lines.append("-" * 40)
    if network_stages:
        for stage in sorted(network_stages.keys()):
            s = network_stages[stage]
            lines.append(f"  {stage}:  nodes={s.get('nodes', 0)}, edges={s.get('edges', 0)}")
    else:
        lines.append("  No network cleaning applied.")

    lines.append("")
    lines.append("-" * 40)
    lines.append("CG OPTIMIZATION")
    lines.append("-" * 40)
    if cg_results:
        lines.append(f"  Configs evaluated:  {cg_results.get('num_configs', 'N/A')}")
        lines.append(f"  Best placement:     {cg_results.get('best_chargers', 'N/A')}")
        obj = cg_results.get('best_objective', 'N/A')
        lines.append(f"  Best objective:     {obj:.4f}" if isinstance(obj, (int, float)) else f"  Best objective:     {obj}")
        lines.append("  All rankings:")
        for cfg in cg_results.get('all_configs', []):
            cfg_str = str(cfg['chargers'])
            lines.append(f"    {cfg_str:>12s}  →  {cfg['objective']:.4f}")
    else:
        lines.append("  (skipped)")

    lines.append("")
    lines.append("-" * 40)
    lines.append("NE CONVERGENCE")
    lines.append("-" * 40)
    if convergence_data:
        for config_str, diffs in convergence_data.items():
            n_iters = len(diffs)
            final = diffs[-1] if diffs else 0
            lines.append(f"  {config_str:>20s}:  {n_iters:>3d} iters, final diff = {final:.1f}")
    else:
        lines.append("  (skipped)")

    lines.append("")
    lines.append("-" * 40)
    lines.append("QUEUE COMPARISON")
    lines.append("-" * 40)
    if queue_results:
        qc = queue_results.get('config', {})
        lines.append(f"  K-routes:   {qc.get('K', 'N/A')}")
        lines.append(f"  MC reps:    {qc.get('N', 'N/A')}")
        lines.append(f"  Single-swap: {qc.get('single_swap', 'N/A')}")
        lines.append(f"  Greedy best:    {queue_results['best_greedy']['positions']}  "
                     f"TT = {queue_results['best_greedy']['avg_travel_time']:.1f}")
        lines.append(f"  Exhaustive best: {queue_results['best_exhaustive']['positions']}  "
                     f"TT = {queue_results['best_exhaustive']['avg_travel_time']:.1f}")
        lines.append(f"  Suboptimality:   {queue_results['suboptimality_pct']:.2f}%")
        lines.append("  All exhaustive:")
        for r in queue_results.get('exhaustive_results', []):
            lines.append(f"    {r['positions']}  →  TT = {r['avg_travel_time']:.1f}")
    else:
        lines.append("  (skipped)")

    if cg_results and queue_results:
        cg_best = cg_results.get('best_chargers')
        q_best = queue_results['best_exhaustive']['positions']
        cg_set = set(int(x) for x in cg_best) if cg_best is not None else set()
        agree = "YES" if cg_set == set(q_best) else "NO"
        lines.append(f"  CG-Queue agree:   {agree}")

    lines.append("")
    lines.append("-" * 40)
    lines.append("TIMING")
    lines.append("-" * 40)
    total = timing.get('total', sum(v for k, v in timing.items() if k != 'total'))
    for step, dur in timing.items():
        if step == 'total':
            continue
        pct = (dur / total * 100) if total > 0 else 0
        lines.append(f"  {step:>22s}: {dur:>6.1f}s ({pct:>5.1f}%)")
    lines.append(f"  {'TOTAL':>22s}: {total:>6.1f}s")

    lines.append("")
    path = os.path.join(experiment_dir, 'run_summary.txt')
    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Run summary saved to {path}")
    return path


def generate_report(experiment_dir, config, timing, cg_results, queue_results, convergence_data=None, network_stages=None):
    """Generate a comprehensive markdown report with all results for comparison."""
    report_path = os.path.join(experiment_dir, "report.md")
    lines = [
        "# EV Charger Optimization Experiment Report",
        "",
        f"**Experiment directory:** `{experiment_dir}`",
        f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Platform:** {platform.system()} ({'queue sim enabled' if QUEUE_SIM_AVAILABLE else 'queue sim skipped'})",
        "",
        "## Configuration",
        "",
        f"| Parameter | Value |",
        f"|---|---|",
        f"| Coordinates | {config.coordinates} |",
        f"| Number of chargers | {config.num_chargers} |",
        f"| Candidate positions | {config.possible_charger_positions} |",
        f"| OD demand | {config.od_demand} |",
        f"| CVXPY solver | {config.use_cvxpy} |",
        f"| CG single-swap | {config.single_swap} |",
        f"| Queue K (routes) | {config.get_queue_param('K')} |",
        f"| Queue alpha | {config.get_queue_param('ALPHA', 0.01)} |",
        f"| Queue NUM_ITERS | {config.get_queue_param('NUM_ITERS')} |",
        f"| Queue N (MC reps) | {config.get_queue_param('N')} |",
        f"| Queue single-swap | {config.get_queue_param('single_swap')} |",
    ]

    if network_stages:
        lines.extend([
            "",
            "## Network Cleaning Stages",
            "",
            "| Stage | Nodes | Edges | Reduction |",
            "|---|---|---|---|",
        ])
        prev_n = None
        for stage_name in sorted(network_stages.keys()):
            info = network_stages[stage_name]
            n, e = info.get('nodes', 0), info.get('edges', 0)
            pct = f"{(1 - n/prev_n)*100:.1f}%" if prev_n and prev_n > 0 else "-"
            lines.append(f"| {stage_name} | {n} | {e} | {pct} |")
            prev_n = n

    lines.extend([
        "",
        "## Timing Breakdown",
        "",
        f"| Step | Wall-clock (s) | % of total |",
        f"|---|---|---|",
    ])
    total = timing.get('total', sum(v for k, v in timing.items() if k != 'total'))
    for step, dur in timing.items():
        if step == 'total':
            continue
        pct = (dur / total * 100) if total > 0 else 0
        lines.append(f"| {step} | {dur:.1f} | {pct:.1f}% |")
    lines.append(f"| **Total** | **{total:.1f}** | 100% |")

    if cg_results:
        lines.extend([
            "",
            "## Congestion-Game Results",
            "",
            f"- Configurations evaluated: {cg_results.get('num_configs', 'N/A')}",
            f"- Best placement: {cg_results.get('best_chargers', 'N/A')}",
            f"- Best objective (total delay): {cg_results.get('best_objective', 'N/A'):.4f}" if isinstance(cg_results.get('best_objective'), (int, float)) else f"- Best objective: {cg_results.get('best_objective', 'N/A')}",
            "",
            "### All CG Configurations",
            "",
            f"| Config | Objective (total delay) |",
            f"|---|---|",
        ])
        bpr_provenance = cg_results.get('bpr_provenance', {})
        if bpr_provenance:
            lines.extend([
                "",
                f"- CG BPR policy: {bpr_provenance.get('policy', 'N/A')}",
                f"- Active BPR status counts: {bpr_provenance.get('fit_status_counts', {})}",
                f"- Active observation sources: {bpr_provenance.get('observation_source_counts', {})}",
                f"- Degraded active links: {bpr_provenance.get('degraded_link_ids', [])}",
            ])
        for cfg_entry in cg_results.get('all_configs', []):
            lines.append(f"| {cfg_entry['chargers']} | {cfg_entry['objective']:.4f} |")

    if convergence_data:
        lines.extend([
            "",
            "## Queue NE Convergence",
            "",
            f"| Config | Iterations | Final diff |",
            f"|---|---|---|",
        ])
        for config_str, diffs in convergence_data.items():
            n_iters = len(diffs)
            final_diff = diffs[-1] if diffs else 0
            lines.append(f"| {config_str} | {n_iters} | {final_diff:.1f} |")

    if queue_results:
        lines.extend([
            "",
            "## Queue-Based Simulation Results",
            "",
            "### Greedy",
            "",
            f"| Placement | Avg travel time |",
            f"|---|---|",
        ])
        for r in queue_results.get('greedy_results', []):
            lines.append(f"| {r['positions']} | {r['avg_travel_time']:.1f} |")

        lines.extend([
            "",
            "### Exhaustive",
            "",
            f"| Placement | Avg travel time |",
            f"|---|---|",
        ])
        for r in queue_results.get('exhaustive_results', []):
            lines.append(f"| {r['positions']} | {r['avg_travel_time']:.1f} |")

        lines.extend([
            "",
            "### Summary",
            "",
            f"- Best greedy: {queue_results['best_greedy']['positions']} (avg TT = {queue_results['best_greedy']['avg_travel_time']:.1f})",
            f"- Best exhaustive: {queue_results['best_exhaustive']['positions']} (avg TT = {queue_results['best_exhaustive']['avg_travel_time']:.1f})",
            f"- Greedy suboptimality: {queue_results['suboptimality_pct']:.2f}%",
            f"- Monte Carlo reps: {queue_results['config']['N']}",
            f"- K (routes): {queue_results['config']['K']}",
            f"- Single swap: {queue_results['config']['single_swap']}",
        ])
    elif not QUEUE_SIM_AVAILABLE:
        lines.extend([
            "",
            "## Queue-Based Simulation",
            "",
            f"**SKIPPED** — queue simulation requires macOS (liblsp.dylib).",
            f"Error: {_QUEUE_SIM_ERROR}",
        ])

    if cg_results and queue_results:
        cg_best = cg_results.get('best_chargers')
        q_greedy = queue_results['best_greedy']['positions']
        q_exhaustive = queue_results['best_exhaustive']['positions']
        lines.extend([
            "",
            "## CG vs Queue Comparison",
            "",
            f"| Model | Best placement | Objective |",
            f"|---|---|---|",
            f"| Congestion game | {cg_best} | {cg_results.get('best_objective', 'N/A'):.4f}" if isinstance(cg_results.get('best_objective'), (int, float)) else f"| Congestion game | {cg_best} | N/A |",
            f"| Queue greedy | {q_greedy} | {queue_results['best_greedy']['avg_travel_time']:.1f} |",
            f"| Queue exhaustive | {q_exhaustive} | {queue_results['best_exhaustive']['avg_travel_time']:.1f} |",
            "",
        ])
        cg_best_set = set(int(x) for x in cg_best) if cg_best is not None else set()
        q_best_set = set(q_exhaustive)
        match = "YES" if cg_best_set == q_best_set else "NO"
        lines.append(f"CG and queue agree on optimal placement: **{match}**")

    lines.extend([
        "",
        "## Intermediate Artifacts",
        "",
        f"| File | Description |",
        f"|---|---|",
        f"| `run_summary.txt` | Unified plain-text summary (all sections) |",
        f"| `run_config.json` | Configuration used for this run |",
        f"| `all_optimization_results.pkl` | CG equilibrium: link flows, route reconstruction, per-config results |",
        f"| `config_*/flow_heatmap.png` | Per-config CG flow heatmaps |",
        f"| `config_*/reconstruction/` | Per-config route reconstruction analysis |",
        f"| `plots/pruning_phases.png` | Network cleaning: nodes/edges per phase + map grid |",
        f"| `plots/ne_convergence.png` | NE convergence: diff vs iteration per config |",
        f"| `plots/timing_breakdown.png` | Pipeline step durations |",
        f"| `plots/bpr_fit_samples.png` | BPR fit diagnostics (random + worst-R² links) |",
        f"| `plots/objective_comparison.png` | CG vs Queue normalized objective bar chart |",
    ])
    if queue_results:
        lines.extend([
            f"| `queue/NE_path_assignments.pkl` | Queue NE route counts per config |",
            f"| `queue/comparison_results.json` | Greedy vs exhaustive results |",
            f"| `queue/ne_convergence.csv` | Per-config, per-iteration NE diff |",
            f"| `queue/traffic_inputs_*.csv` | Network CSVs for simulator |",
            f"| `queue/traffic_outputs/` | Simulator raw stats (t_stats, link_stats, node_stats) |",
        ])
    lines.extend([
        f"| `experiment_summary.json` | Machine-readable summary of all results |",
        f"| `report.md` | This report |",
        "",
    ])

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Report saved to {report_path}")
    return report_path


def run_pipeline(config_path: str, results_root: str = "results", resume: bool = False) -> str:
    """Run the complete EV charger optimization pipeline end-to-end.

    Returns: path to experiment directory.
    """
    t_start = time.time()
    config = Config.from_json(config_path)
    seed_manager = SeedManager(config.pipeline.get('random_seed', 0))
    timing_recorder = TimingRecorder()

    digest = config_digest(config.to_dict())
    experiment_dir = os.path.join(results_root, f"{safe_name(config.name)}-{digest[:12]}")
    if os.path.exists(experiment_dir) and not resume:
        raise FileExistsError(
            f"Run directory already exists: {experiment_dir}; use --resume or change the config"
        )
    os.makedirs(experiment_dir, exist_ok=True)
    with open(config_path) as handle:
        original_config = json.load(handle)
    atomic_write_json(os.path.join(experiment_dir, "original_config.json"), original_config)
    config.to_json(os.path.join(experiment_dir, "run_config.json"))
    atomic_write_json(os.path.join(experiment_dir, "status.json"), {
        "status": "running", "stage": "network", "config_digest": digest,
        "resume": bool(resume), **process_provenance(),
    })

    timing = {}
    cg_results = None
    queue_results = None
    network_stages = None
    network_stage_maps = None
    network_node_count = None
    network_edge_count = None
    network_manifest = None
    shared_road_net = None
    solver_metadata = {
        'mode': 'cvxpy' if config.use_cvxpy else 'scipy',
    }
    if config.use_cvxpy:
        try:
            import cvxpy as cp
            solver_metadata['cvxpy_version'] = cp.__version__
            solver_metadata['installed_solvers'] = cp.installed_solvers()
        except Exception as exc:
            solver_metadata['error'] = str(exc)

    # Step 0: Network download + cleaning (shared across all steps)
    print("\n" + "=" * 80)
    print("STEP 0: Network Download + Cleaning")
    print("=" * 80)
    t0 = time.time()
    rf = config.road_filter
    hw_types = rf.get('highway_types') if rf.get('enabled', True) else None
    from src.road_network import RoadNet
    shared_road_net = RoadNet('pipeline')
    input_artifact_dir = config.pipeline.get('artifact_dir')
    input_network_manifest = None
    if input_artifact_dir:
        input_network_manifest = shared_road_net.load_artifact(input_artifact_dir)
    else:
        shared_road_net.get_map(
            config.coordinates[0], config.coordinates[1],
            config.coordinates[2], config.coordinates[3],
            highway_types=hw_types,
            merge_chains=rf.get('merge_chains', True),
            contract_threshold=rf.get('contract_threshold', 30),
            prune_dead_ends=rf.get('prune_dead_ends', False),
            suppress_t_junctions=rf.get('suppress_t_junctions', False),
            apply_cleaning=rf.get('enabled', True),
            intersection_tolerance=rf.get('intersection_tolerance', 0),
            cache_policy=config.network.get('cache_policy', 'reuse'),
        )
    network_stages = shared_road_net.stage_counts
    network_stage_maps = shared_road_net.stage_maps
    network_node_count = len(shared_road_net.nodes)
    network_edge_count = len(shared_road_net.edges)
    expected_nodes = config.network.get('expected_nodes')
    if expected_nodes is not None:
        tolerance = float(config.network.get('node_tolerance_fraction', 0.10))
        relative_error = abs(network_node_count - int(expected_nodes)) / int(expected_nodes)
        if relative_error > tolerance:
            atomic_write_json(os.path.join(experiment_dir, "status.json"), {
                "status": "ineligible", "stage": "network",
                "reason": "network_size_out_of_tolerance",
                "expected_nodes": int(expected_nodes), "actual_nodes": network_node_count,
                "relative_error": relative_error, "allowed_relative_error": tolerance,
                "config_digest": digest, **process_provenance(),
            })
            raise ValueError(
                f"Generated network has {network_node_count} nodes; expected "
                f"{expected_nodes} within {tolerance:.1%}"
            )
    print(f"Network: {network_node_count} nodes, {network_edge_count} links")
    timing['network_cleaning'] = time.time() - t0
    timing_recorder.add('network_cleaning', timing['network_cleaning'], nodes=network_node_count, edges=network_edge_count)

    network_artifact_dir = os.path.join(experiment_dir, 'network')
    network_manifest = shared_road_net.export_artifact(
        network_artifact_dir,
        source={
            'input_artifact': input_artifact_dir,
            'input_network_hash': (
                input_network_manifest.get('network_hash')
                if input_network_manifest else None
            ),
            'coordinates': config.coordinates,
            'highway_types': hw_types,
            'merge_chains': rf.get('merge_chains', True),
            'contract_threshold': rf.get('contract_threshold', 30),
            'intersection_tolerance': rf.get('intersection_tolerance', 0),
            'prune_dead_ends': rf.get('prune_dead_ends', False),
            'random_seed': seed_manager.seed,
        },
    )
    with open(os.path.join(experiment_dir, 'network_manifest.json'), 'w') as handle:
        json.dump(network_manifest, handle, indent=2, default=str)

    scenario_metadata = None
    if config.scenario_generation.get("enabled", False):
        from src.scenario_generation import generate_scenario, plot_scenario
        generated = generate_scenario(shared_road_net, config.scenario_generation)
        config.possible_charger_positions = generated.candidate_node_ids
        config.num_chargers = int(config.scenario_generation["num_chargers"])
        config.od_demand = generated.od_demand
        scenario_metadata = generated.metadata
        resolved = config.to_dict()
        resolved["generated_scenario"] = scenario_metadata
        resolved["network_hash"] = network_manifest["network_hash"]
        atomic_write_json(os.path.join(experiment_dir, "resolved_config.json"), resolved)
    else:
        atomic_write_json(os.path.join(experiment_dir, "resolved_config.json"), {
            **config.to_dict(), "network_hash": network_manifest["network_hash"],
            "generated_scenario": None,
        })
    known_nodes = set(int(value) for value in shared_road_net.nodes['node_id'])
    invalid_candidates = sorted(set(config.possible_charger_positions) - known_nodes)
    invalid_od = sorted({
        node for record in config.get_demand_classes()
        for node in (record.origin, record.destination)
        if node not in known_nodes
    })
    if invalid_candidates or invalid_od:
        raise ValueError(
            f'Configuration references nodes absent from canonical network: '
            f'candidates={invalid_candidates}, od_nodes={invalid_od}'
        )

    plot_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    _plot_pruning_phases(network_stages, os.path.join(plot_dir, 'pruning_phases.png'),
                         node_count=network_node_count, edge_count=network_edge_count,
                         stage_maps=network_stage_maps)
    if scenario_metadata is not None:
        plot_scenario(
            shared_road_net, generated, os.path.join(plot_dir, "generated_scenario.png")
        )
    atomic_write_json(os.path.join(experiment_dir, "status.json"), {
        "status": "running", "stage": "bpr", "config_digest": digest,
        "network_hash": network_manifest["network_hash"], **process_provenance(),
    })

    # Step 1: BPR fitting
    bpr_config = dict(config.pipeline.get("bpr_generation", {}))
    global_parallel_workers = config.pipeline.get("parallel_workers")
    if global_parallel_workers is not None:
        if bpr_config.get('workers') is None:
            bpr_config['workers'] = int(global_parallel_workers)
        if bpr_config.get('fit_workers') is None:
            bpr_config['fit_workers'] = int(global_parallel_workers)
    rf = config.road_filter
    # BPR caches are keyed by the exact canonical network, not a display name.
    network_hash = network_manifest['network_hash']
    bpr_dir = os.path.join(experiment_dir, 'bpr')
    os.makedirs(bpr_dir, exist_ok=True)
    bpr_data_path = os.path.join(bpr_dir, 'traffic_data.csv')
    bpr_cache_path = os.path.join(bpr_dir, 'cached_results.pkl')
    # Fall back to original cache ONLY when no topology changes
    if not os.path.exists(bpr_cache_path) and not os.path.exists(bpr_data_path):
        # Read-only compatibility with old project caches. They are accepted
        # only when their link coverage matches the canonical graph below.
        if os.path.exists("data/cached_results.pkl"):
            bpr_cache_path = "data/cached_results.pkl"
        elif os.path.exists("data/traffic_data.csv"):
            bpr_data_path = "data/traffic_data.csv"

    if not config.pipeline.get("skip_bpr_fitting", False):
        print("\n" + "=" * 80)
        print("STEP 1: BPR Fitting")
        print("=" * 80)
        t0 = time.time()
        pandas_df, model_fitter = load_or_fit_model(
            data_path=bpr_data_path,
            cache_path=bpr_cache_path,
            coordinates=config.coordinates,
            bpr_config=bpr_config,
            road_filter_config=config.road_filter,
            road_net=shared_road_net,
            artifact_dir=network_artifact_dir,
            n_links=network_edge_count,
            work_dir=bpr_dir,
            seed_manager=seed_manager,
        )
        timing['bpr_fitting'] = time.time() - t0
        timing_recorder.add('bpr_fitting', timing['bpr_fitting'], links=network_edge_count)
    else:
        print("\nSkipping BPR fitting (using cache).")
        t0 = time.time()
        pandas_df, model_fitter = load_or_fit_model(
            data_path=bpr_data_path,
            cache_path=bpr_cache_path,
            coordinates=config.coordinates,
            bpr_config=bpr_config,
            road_filter_config=config.road_filter,
            road_net=shared_road_net,
            artifact_dir=network_artifact_dir,
            n_links=network_edge_count,
            work_dir=bpr_dir,
            seed_manager=seed_manager,
            allow_generate=False,
        )
        timing['bpr_fitting'] = time.time() - t0
        timing_recorder.add('bpr_fitting', timing['bpr_fitting'], skipped=True)

    try:
        _plot_bpr_fit_samples(
            pandas_df, os.path.join(plot_dir, 'bpr_fit_samples.png'),
            seed=seed_manager.seed,
        )
        _plot_historical_bpr_comparison(
            pandas_df,
            os.path.join(plot_dir, 'bpr_historical_comparison.png'),
            reference_commit=bpr_config.get('historical_reference_commit', '37eab33'),
        )
    except Exception:
        pass

    # Step 2: Congestion-game equilibrium
    if not config.pipeline.get("skip_cg_optimization", False):
        print("\n" + "=" * 80)
        print("STEP 2: Congestion-Game Equilibrium")
        print("=" * 80)
        t0 = time.time()
        od_demand = config.get_od_demand_tuples()
        grids, time_history, experiment_dir = outer_optimization(
            coordinates=config.coordinates,
            num_chargers=config.num_chargers,
            possible_charger_positions=config.possible_charger_positions,
            calculate_on_all_possible_positions=config.calculate_on_all_possible_positions,
            parameter_fit_results=pandas_df,
            max_iter=config.max_iter,
            use_derivatives=config.use_derivatives,
            single_swap=config.single_swap,
            use_cvxpy=config.use_cvxpy,
            od_demand=od_demand,
            plot_info=config.plot_info,
            config_filepath=config_path,
            output_dir=experiment_dir,
            road_net=shared_road_net,
            charger_self_link_length=config.charger_self_link_length,
            cg_fit_policy=config.pipeline.get('cg_fit_policy', 'allow_degraded'),
            parallel_workers=(
                config.pipeline.get('parallel_workers') or available_cpus()
            ),
            checkpoint_dir=os.path.join(experiment_dir, 'cg_checkpoints'),
            resume=resume,
        )
        if not grids or not any(np.isfinite(float(grid.travel_time_obj)) for grid in grids):
            raise RuntimeError(
                'Congestion-game optimization produced no finite placement objective; '
                f'solver metadata: {solver_metadata}'
            )
        timing['cg_optimization'] = time.time() - t0
        timing_recorder.add('cg_optimization', timing['cg_optimization'], configurations=len(grids))
        best_grid = grids[np.argmin([g.travel_time_obj for g in grids])]
        cg_results = {
            'best_chargers': [int(x) for x in best_grid.chargers],
            'best_objective': float(best_grid.travel_time_obj),
            'num_configs': len(grids),
            'all_configs': [
                {
                    'chargers': [int(x) for x in g.chargers],
                    'objective': float(g.travel_time_obj),
                    'solver': getattr(g, 'solver_metadata', {}),
                }
                for g in grids
            ],
            'stage_counts': grids[0].net.stage_counts if hasattr(grids[0], 'net') else {},
            'bpr_provenance': getattr(best_grid, 'bpr_provenance', {}),
        }
        all_opt_path = os.path.join(experiment_dir, 'all_optimization_results.pkl')
        grid_solver_metadata = getattr(
            best_grid, 'solver_metadata', getattr(best_grid.net, 'solver_metadata', {})
        )
        solver_metadata.update(grid_solver_metadata)
        if os.path.exists(all_opt_path):
            # Attach the exact artifact identity and solver provenance to the
            # object consumed by route recovery and queue stages.
            with open(all_opt_path, 'rb') as handle:
                optimization_data = pickle.load(handle)
            run_configuration = dict(optimization_data.get('run_configuration', {}))
            run_configuration.update({
                'network_hash': network_hash,
                'network_artifact': os.path.relpath(network_artifact_dir, experiment_dir),
                'random_seed': seed_manager.seed,
                'solver': grid_solver_metadata,
            })
            optimization_data['run_configuration'] = run_configuration
            optimization_data['network_hash'] = network_hash
            with open(all_opt_path, 'wb') as handle:
                pickle.dump(optimization_data, handle)
    else:
        print("\nSkipping CG optimization.")
        timing['cg_optimization'] = 0
        all_opt_path = os.path.join(experiment_dir, 'all_optimization_results.pkl')
        if not os.path.exists(all_opt_path):
            raise FileNotFoundError(f"CG optimization skipped but {all_opt_path} not found.")

    # Step 3 & 4: Queue-based simulation
    convergence_data = None
    queue_enabled = config.queue_simulation.get('enabled', True)
    if (not config.pipeline.get("skip_queue_simulation", False)
            and queue_enabled and QUEUE_SIM_AVAILABLE):
        print("\n" + "=" * 80)
        print("STEP 3: Queue-Based NE Assignments")
        print("=" * 80)
        from queue_sim.find_nash import find_nash_assignments
        t0 = time.time()
        ne_pkl_path, convergence_data = find_nash_assignments(
            config, experiment_dir, all_opt_path,
            artifact_dir=network_artifact_dir,
            seed_manager=seed_manager,
        )
        timing['queue_ne'] = time.time() - t0
        timing_recorder.add('queue_ne', timing['queue_ne'])

        queue_manifest_path = os.path.join(experiment_dir, 'queue', 'queue_manifest.json')
        with open(queue_manifest_path) as handle:
            queue_manifest = json.load(handle)
        if queue_manifest.get('nonconverged_configurations'):
            raise RuntimeError(
                'Queue better-response search did not converge; comparison is '
                f'not eligible: {queue_manifest["nonconverged_configurations"]}'
            )

        if convergence_data:
            _save_convergence_csv(convergence_data, os.path.join(experiment_dir, 'queue', 'ne_convergence.csv'))
            print(f"Convergence data saved to {experiment_dir}/queue/ne_convergence.csv")
            _plot_ne_convergence(convergence_data, os.path.join(plot_dir, 'ne_convergence.png'))

        print("\n" + "=" * 80)
        print("STEP 4: Queue-Based Greedy vs Exhaustive Comparison")
        print("=" * 80)
        from queue_sim.comparison import run_comparison
        t0 = time.time()
        queue_results = run_comparison(
            config, experiment_dir, all_opt_path, ne_pkl_path,
            artifact_dir=network_artifact_dir,
            seed_manager=seed_manager,
        )
        timing['queue_comparison'] = time.time() - t0
        timing_recorder.add('queue_comparison', timing['queue_comparison'])

        if cg_results and queue_results:
            _plot_objective_comparison(cg_results, queue_results,
                                       os.path.join(plot_dir, 'objective_comparison.png'))
    elif (not config.pipeline.get("skip_queue_simulation", False)
          and queue_enabled and not QUEUE_SIM_AVAILABLE):
        raise RuntimeError(
            "Queue simulation is required by this configuration but its "
            f"native library is unavailable: {_QUEUE_SIM_ERROR}"
        )
    else:
        print("\nSkipping queue simulation (config).")
        timing['queue_ne'] = 0
        timing['queue_comparison'] = 0

    # Step 5: Aggregate summaries
    print("\n" + "=" * 80)
    print("STEP 5: Summaries")
    print("=" * 80)
    timing['total'] = time.time() - t_start

    timing_plot_data = {k: v for k, v in timing.items() if k != 'total'}
    if timing_plot_data:
        _plot_timing_breakdown(timing_plot_data, os.path.join(plot_dir, 'timing_breakdown.png'))

    _generate_run_summary(experiment_dir, config, timing, cg_results, queue_results,
                          convergence_data, network_stages)
    generate_report(experiment_dir, config, timing, cg_results, queue_results,
                    convergence_data, network_stages)

    provenance = {
        'random_seed': seed_manager.seed,
        'parallel_workers_requested': config.pipeline.get('parallel_workers'),
        'parallel_workers_available': available_cpus(),
        'network_hash': network_hash,
        'network_artifact': os.path.relpath(network_artifact_dir, experiment_dir),
        'config_path': os.path.abspath(config_path),
        'python': sys.version,
        'platform': platform.platform(),
        'solver': solver_metadata,
        'cg_fit_policy': config.pipeline.get('cg_fit_policy', 'allow_degraded'),
        'cg_bpr_provenance': cg_results.get('bpr_provenance', {}) if cg_results else {},
        'config_digest': digest,
        'graph_cache': getattr(shared_road_net, 'cache_metadata', {}),
        'generated_scenario': scenario_metadata,
    }
    with open(os.path.join(experiment_dir, 'run_manifest.json'), 'w') as handle:
        json.dump({
            'provenance': provenance,
            'timing_events': timing_recorder.events,
            'timing': timing,
            'network': network_manifest,
        }, handle, indent=2, default=str)

    summary = {
        'experiment_dir': experiment_dir,
        'timestamp': datetime.now().isoformat(),
        'platform': platform.system(),
        'queue_sim_available': QUEUE_SIM_AVAILABLE,
        'config': config.to_dict(),
        'timing': timing,
        'cg_results': cg_results,
        'queue_results': queue_results,
        'ne_convergence': {k: v for k, v in convergence_data.items()} if convergence_data else None,
        'network_stages': network_stages,
        'provenance': provenance,
        'timing_events': timing_recorder.events,
    }
    summary_path = os.path.join(experiment_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Experiment summary saved to {summary_path}")

    # Validate the complete cross-stage artifact contract before declaring the
    # run successful.  Queue validation is required only when that stage
    # actually produced results; it remains optional for environments where
    # the simulator is unavailable or explicitly disabled.
    validation = validate_experiment_outputs(
        experiment_dir,
        require_cg=not config.pipeline.get("skip_cg_optimization", False),
        require_queue=queue_results is not None,
    )
    with open(os.path.join(experiment_dir, 'sanity_check.json'), 'w') as handle:
        json.dump(validation, handle, indent=2, default=str)
    if not validation['valid']:
        raise RuntimeError(
            'Experiment output sanity checks failed; see '
            f"{os.path.join(experiment_dir, 'sanity_check.json')}: "
            + '; '.join(validation['errors'])
        )
    print(f"Sanity checks passed: {os.path.join(experiment_dir, 'sanity_check.json')}")

    inventory = directory_inventory(experiment_dir)
    atomic_write_json(os.path.join(experiment_dir, "artifact_inventory.json"), inventory)
    bpr_manifest_path = os.path.join(experiment_dir, 'bpr', 'bpr_manifest.json')
    bpr_manifest = {}
    if os.path.isfile(bpr_manifest_path):
        with open(bpr_manifest_path) as handle:
            bpr_manifest = json.load(handle)
    summary_row = {
        'run_id': os.path.basename(experiment_dir),
        'status': 'complete',
        'eligible': True,
        'network_hash': network_hash,
        'nodes': network_node_count,
        'edges': network_edge_count,
        'bpr_fit_status_counts': json.dumps(
            bpr_manifest.get('fit_status_counts', {}), sort_keys=True
        ),
        'cg_configurations': len(grids) if not config.pipeline.get(
            'skip_cg_optimization', False
        ) else 0,
        'queue_status': (
            queue_results.get('status', 'complete') if queue_results else 'skipped'
        ),
        'network_seconds': timing.get('network_cleaning'),
        'bpr_seconds': timing.get('bpr_fitting'),
        'cg_seconds': timing.get('cg_optimization'),
        'queue_ne_seconds': timing.get('queue_ne'),
        'queue_comparison_seconds': timing.get('queue_comparison'),
        'total_seconds': timing.get('total'),
        'artifact_bytes': inventory['total_bytes'],
        'available_cpus': available_cpus(),
    }
    pd.DataFrame([summary_row]).to_csv(
        os.path.join(experiment_dir, 'summary.csv'), index=False
    )
    run_manifest_path = os.path.join(experiment_dir, 'run_manifest.json')
    with open(run_manifest_path) as handle:
        completed_manifest = json.load(handle)
    completed_manifest['artifacts'] = inventory
    atomic_write_json(run_manifest_path, completed_manifest)
    atomic_write_json(os.path.join(experiment_dir, "status.json"), {
        "status": "complete", "stage": "complete", "eligible": True,
        "config_digest": digest, "network_hash": network_hash,
        "timing": timing, "artifact_bytes": inventory["total_bytes"],
        **process_provenance(),
    })

    print(f"\n{'=' * 80}")
    print(f"Pipeline complete. Experiment directory: {experiment_dir}")
    print(f"Total time: {timing['total']:.1f}s")
    print(f"{'=' * 80}")

    return experiment_dir


def run_network_only(config_path: str) -> str:
    """Download + clean network only, generate pruning plot, exit. For rapid iteration."""
    config = NetworkConfig.from_json(config_path)
    seed_manager = SeedManager(config.road_filter.get('diagnostic_seed', 0))
    rf = config.road_filter
    coords = config.coordinates
    from src.network_pruning import ROAD_PROFILES
    highway_types = rf.get('highway_types')
    if highway_types is None and rf.get('enabled', True):
        highway_types = list(ROAD_PROFILES[rf.get('road_profile', 'secondary_plus')])
    if not rf.get('enabled', True):
        highway_types = None
    do_merge = rf.get('merge_chains', True)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    root = config.output_dir or "results"
    exp_dir = os.path.join(root, f"{timestamp}_{config.name}_network_only")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "run_config.json"), "w") as handle:
        json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
    plot_dir = os.path.join(exp_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    from src.road_network import RoadNet
    t0 = time.time()
    rn = RoadNet('network_only')
    rn.get_map(coords[0], coords[1], coords[2], coords[3],
               highway_types=highway_types, merge_chains=do_merge,
               contract_threshold=rf.get('contract_threshold', 30),
               prune_dead_ends=rf.get('prune_dead_ends', False),
               suppress_t_junctions=rf.get('suppress_t_junctions', False),
               apply_cleaning=rf.get('enabled', True),
               intersection_tolerance=rf.get('intersection_tolerance', 0))
    elapsed = time.time() - t0
    artifact_dir = os.path.join(exp_dir, 'network')
    manifest = rn.export_artifact(
        artifact_dir,
        source={
            'coordinates': coords,
            'highway_types': highway_types,
            'merge_chains': do_merge,
            'contract_threshold': rf.get('contract_threshold', 30),
            'intersection_tolerance': rf.get('intersection_tolerance', 0),
            'prune_dead_ends': rf.get('prune_dead_ends', False),
            'random_seed': seed_manager.seed,
        },
    )
    with open(os.path.join(exp_dir, 'run_manifest.json'), 'w') as handle:
        json.dump({'network': manifest, 'timing': {'network_cleaning': elapsed}}, handle, indent=2)
    with open(os.path.join(exp_dir, 'network_manifest.json'), 'w') as handle:
        json.dump(manifest, handle, indent=2, default=str)

    print(f"\nNetwork download + cleaning: {elapsed:.1f}s")
    for k, v in sorted(rn.stage_counts.items()):
        print(f"  {k}:  nodes={v['nodes']:>4d}, edges={v['edges']:>4d}")

    _plot_pruning_phases(rn.stage_counts, os.path.join(plot_dir, 'pruning_phases.png'),
                         stage_maps=rn.stage_maps)
    validation = validate_experiment_outputs(
        exp_dir,
        require_cg=False,
        require_queue=False,
        require_reports=False,
    )
    with open(os.path.join(exp_dir, 'sanity_check.json'), 'w') as handle:
        json.dump(validation, handle, indent=2, default=str)
    if not validation['valid']:
        raise RuntimeError(
            'Network-only output sanity checks failed: '
            + '; '.join(validation['errors'])
        )
    print(f"Pruning plot saved to {plot_dir}/pruning_phases.png")
    print(f"Canonical network artifact saved to {artifact_dir} (hash={manifest['network_hash']})")
    return exp_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run unified EV charger optimization pipeline')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config JSON')
    parser.add_argument('--results-root', default='results', help='Root directory for deterministic run outputs')
    parser.add_argument('--resume', action='store_true', help='Resume/reuse checkpoints in the deterministic run directory')
    parser.add_argument('--validate-config', action='store_true', help='Validate and print the normalized config, then exit')
    parser.add_argument('--network-only', action='store_true',
                        help='Download + clean network, generate pruning plot, exit')
    parser.add_argument('--pruning-sweep', action='store_true',
                        help='Compare road profiles and intersection radii; run no optimization')
    args = parser.parse_args()
    if args.validate_config:
        validated = Config.from_json(args.config)
        print(json.dumps(validated.to_dict(), indent=2, sort_keys=True))
        raise SystemExit(0)
    if args.network_only and args.pruning_sweep:
        parser.error('--network-only and --pruning-sweep are mutually exclusive')
    if args.pruning_sweep:
        from src.pruning_study import run_pruning_sweep
        run_pruning_sweep(args.config)
    elif args.network_only:
        run_network_only(args.config)
    else:
        run_pipeline(args.config, results_root=args.results_root, resume=args.resume)
