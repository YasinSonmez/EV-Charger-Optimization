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
import multiprocessing
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
from matplotlib.lines import Line2D
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config, NetworkConfig
from src.plot_style import (
    COLORS, apply_paper_style, clean_axis, save_publication_figure,
)
from src.contracts import (
    BPR_CALIBRATION_VERSION, SeedManager, TimingRecorder, stable_json,
)
from src.network_artifact import load_network_artifact
from src.run_state import (
    atomic_write_json, available_cpus, config_digest,
    directory_inventory, process_provenance, safe_name,
)
from src.sanity_checks import validate_experiment_outputs
from src.model_fitter import TrafficModelFitter, convert_string_to_array, validate_bpr_fit_table
from src.utils import outer_optimization

apply_paper_style()

try:
    from queue_sim import QUEUE_SIM_AVAILABLE, _QUEUE_SIM_ERROR
except (ImportError, ModuleNotFoundError):
    QUEUE_SIM_AVAILABLE = False
    _QUEUE_SIM_ERROR = "queue_sim package or platform-native library not found"


def _cleanup_multiprocessing_children(timeout_seconds=5.0):
    """Terminate unexpected workers before Python/container shutdown."""
    children = multiprocessing.active_children()
    if not children:
        print("Process shutdown check: no active multiprocessing children")
        return []
    descriptions = [f"pid={child.pid} name={child.name}" for child in children]
    print(
        "WARNING: terminating active multiprocessing children at shutdown: "
        + ", ".join(descriptions)
    )
    for child in children:
        if child.is_alive():
            child.terminate()
    deadline = time.monotonic() + float(timeout_seconds)
    for child in children:
        child.join(max(0.0, deadline - time.monotonic()))
    for child in children:
        if child.is_alive():
            child.kill()
            child.join(1.0)
    return descriptions


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


def _ensure_bpr_reference_metadata(
        pandas_df, artifact_dir, capacity_source='simulator',
        capacity_per_lane=1900.0, calibration_window_hours=0.1):
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
                * float(calibration_window_hours)
            )
        else:
            output['calibration_capacity'] = output['link_id'].map(
                lambda link_id: float(by_id.loc[int(link_id), 'lanes'])
                * float(capacity_per_lane) * float(calibration_window_hours)
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


def _bpr_request_manifest_fields(network_hash, bpr_config, seed):
    """Return the request identity shared by BPR data and fit checkpoints."""
    requested_mode = bpr_config.get('mode', 'historical_artifact_compatible')
    return {
        'network_hash': network_hash,
        'bpr_mode': requested_mode,
        'num_samples': int(bpr_config.get('num_samples', 25)),
        'max_flow': float(bpr_config.get('max_flow', 250)),
        'random_seed': int(seed if seed is not None else 0),
        'fitter_version': (
            'historical_v1'
            if requested_mode == 'historical_artifact_compatible'
            else BPR_CALIBRATION_VERSION
        ),
        'route_semantics': (
            'measured_target_flow_with_straight_ahead_context'
            if requested_mode == 'historical_artifact_compatible'
            else 'offered_cohort_entry_wait_inclusive_nonbinding_continuation'
        ),
        'flow_fractions': (
            [float(value) for value in (bpr_config.get('flow_fractions') or [])]
            if requested_mode == 'capacity_fraction_strict' else None
        ),
        'capacity_source': (
            bpr_config.get('capacity_source', 'simulator')
            if requested_mode == 'capacity_fraction_strict' else None
        ),
        'capacity_per_lane': (
            float(bpr_config.get('capacity_per_lane', 1900.0))
            if requested_mode == 'capacity_fraction_strict' else None
        ),
        'calibration_window_hours': (
            float(bpr_config.get('calibration_window_hours', 0.1))
            if requested_mode == 'capacity_fraction_strict' else None
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
        'probe_continuation_capacity_multiplier': float(
            bpr_config.get('probe_continuation_capacity_multiplier', 10.0)
        ) if requested_mode == 'capacity_fraction_strict' else None,
        'simulation_horizon': int(bpr_config.get('simulation_horizon', 10801)),
    }


def _bpr_manifest_is_compatible(manifest_path, network_hash, bpr_config, seed):
    """Return whether a cached BPR artifact matches the current request."""
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return False
    checks = _bpr_request_manifest_fields(network_hash, bpr_config, seed)
    for key, expected in checks.items():
        actual = manifest.get(key)
        if key in {
            'max_flow', 'correlation_threshold', 'variation_ratio_threshold',
            'synthetic_context_capacity_multiplier', 'synthetic_context_length_m',
            'capacity_per_lane', 'calibration_window_hours',
            'probe_continuation_capacity_multiplier',
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
    # ``cache_path`` may point at a historical repository artifact selected as
    # a read-only compatibility input. Any new or enriched checkpoint belongs
    # in this run's writable BPR directory.
    output_cache_path = (
        os.path.join(work_dir, 'cached_results.pkl') if work_dir else cache_path
    )
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
        pandas_df = pd.read_csv(data_path)
        convert_string_to_array(pandas_df, 'x_vector')
        convert_string_to_array(pandas_df, 'y_vector')
        _validate_bpr_network_rows(pandas_df, network_hash)
        if bpr_mode == 'capacity_fraction_strict':
            pandas_df = _ensure_bpr_reference_metadata(
                pandas_df, artifact_dir,
                capacity_source=bpr_config.get('capacity_source', 'simulator'),
                capacity_per_lane=bpr_config.get('capacity_per_lane', 1900.0),
                calibration_window_hours=bpr_config.get('calibration_window_hours', 0.1),
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
        with open(output_cache_path, "wb") as f:
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
            probe_continuation_capacity_multiplier=bpr_config.get(
                'probe_continuation_capacity_multiplier', 10.0
            ),
            simulation_horizon=bpr_config.get(
                'simulation_horizon',
                10801,
            ),
            active_link_ids=bpr_config.get('active_link_ids'),
            resume=bpr_config.get('resume', True),
        )
        # The per-link simulations are complete now. Record the full request
        # identity before fitting, so an interrupted fit can resume directly
        # from traffic_data.csv without rescheduling simulation work.
        generation_manifest_path = os.path.join(
            work_dir or os.path.dirname(output_cache_path) or '.',
            'bpr_manifest.json',
        )
        generation_manifest = {}
        if os.path.exists(generation_manifest_path):
            with open(generation_manifest_path) as handle:
                generation_manifest = json.load(handle)
        generation_manifest.update(_bpr_request_manifest_fields(
            network_hash,
            bpr_config,
            seed_manager.seed if seed_manager is not None else 0,
        ))
        atomic_write_json(generation_manifest_path, generation_manifest)
        pandas_df = pd.read_csv(data_path)
        convert_string_to_array(pandas_df, 'x_vector')
        convert_string_to_array(pandas_df, 'y_vector')
        _validate_bpr_network_rows(pandas_df, network_hash)
        if bpr_mode == 'capacity_fraction_strict':
            pandas_df = _ensure_bpr_reference_metadata(
                pandas_df, artifact_dir,
                capacity_source=bpr_config.get('capacity_source', 'simulator'),
                capacity_per_lane=bpr_config.get('capacity_per_lane', 1900.0),
                calibration_window_hours=bpr_config.get('calibration_window_hours', 0.1),
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
        with open(output_cache_path, "wb") as f:
            pickle.dump((pandas_df, model_fitter), f)
        print("Generated BPR data and cached fit results.")
    else:
        raise FileNotFoundError(
            f"No BPR data found at {data_path} or {cache_path}. "
            f"Provide coordinates and run where the queue native library is available."
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
        with open(output_cache_path, "wb") as f:
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
                else BPR_CALIBRATION_VERSION
            ),
            'historical_reference_commit': bpr_config.get('historical_reference_commit') if bpr_mode == 'historical_artifact_compatible' else None,
            'route_semantics': (
                'measured_target_flow_with_straight_ahead_context'
                if bpr_mode == 'historical_artifact_compatible'
                else 'offered_cohort_entry_wait_inclusive_nonbinding_continuation'
            ),
            'missing_context_policy': bpr_config.get('missing_context_policy', 'synthetic_boundary'),
            'synthetic_context_capacity_multiplier': float(
                bpr_config.get('synthetic_context_capacity_multiplier', 10.0)
            ),
            'synthetic_context_length_m': float(
                bpr_config.get('synthetic_context_length_m', 1.0)
            ),
            'probe_continuation_capacity_multiplier': float(
                bpr_config.get('probe_continuation_capacity_multiplier', 10.0)
            ) if bpr_mode == 'capacity_fraction_strict' else None,
            'flow_fractions': (
                [float(value) for value in (bpr_config.get('flow_fractions') or [])]
                if bpr_mode == 'capacity_fraction_strict' else None
            ),
            'capacity_source': (
                bpr_config.get('capacity_source', 'simulator')
                if bpr_mode == 'capacity_fraction_strict' else None
            ),
            'capacity_per_lane': (
                float(bpr_config.get('capacity_per_lane', 1900.0))
                if bpr_mode == 'capacity_fraction_strict' else None
            ),
            'calibration_window_hours': (
                float(bpr_config.get('calibration_window_hours', 0.1))
                if bpr_mode == 'capacity_fraction_strict' else None
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
    """Plot paired placement objectives and cross-model agreement."""
    if not cg_results or not queue_results:
        return

    target_size = int(queue_results.get('config', {}).get('num_stations', 0))
    cg_configs = [
        value for value in cg_results.get('all_configs', [])
        if not target_size or len(set(value.get('chargers', []))) == target_size
    ]
    q_results = queue_results.get('exhaustive_results', [])

    cg_map = {
        tuple(sorted(c['chargers'])): float(c['objective']) for c in cg_configs
    }
    q_map = {
        tuple(sorted(r['positions'])): float(r['avg_travel_time'])
        for r in q_results if np.isfinite(r['avg_travel_time'])
    }
    placements = [value for value in cg_map if value in q_map]
    if not placements:
        return
    placements.sort(key=lambda value: cg_map[value])
    cg = np.asarray([cg_map[value] for value in placements], dtype=float)
    queue = np.asarray([q_map[value] for value in placements], dtype=float)
    cg_norm, queue_norm = cg / np.min(cg), queue / np.min(queue)
    labels = ['+'.join(map(str, value)) for value in placements]
    x = np.arange(len(placements))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for index in x:
        ax1.plot(
            [index, index], [cg_norm[index], queue_norm[index]],
            color=COLORS['light'], linewidth=1.2, zorder=1,
        )
    ax1.scatter(x, cg_norm, color=COLORS['blue'], s=35, label='Congestion game', zorder=2)
    ax1.scatter(x, queue_norm, color=COLORS['orange'], marker='s', s=32,
                label='Queue simulation', zorder=2)
    ax1.axhline(1.0, color=COLORS['mid'], linestyle=':', linewidth=0.9)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_ylabel('Objective / model-specific minimum')
    ax1.set_xlabel('Charger placement (ordered by CG objective)')
    ax1.set_title('Relative objective by placement')
    ax1.legend(loc='upper left')
    clean_axis(ax1)

    ax2.scatter(cg_norm, queue_norm, color=COLORS['purple'], s=42,
                edgecolor='white', linewidth=0.5, zorder=2)
    lo = min(np.min(cg_norm), np.min(queue_norm))
    hi = max(np.max(cg_norm), np.max(queue_norm))
    pad = max(0.01, 0.06 * (hi - lo))
    ax2.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=COLORS['mid'],
             linestyle='--', linewidth=0.9, label='Equal relative objective')
    for cx, qx, label in zip(cg_norm, queue_norm, labels):
        ax2.annotate(label, (cx, qx), xytext=(4, 3), textcoords='offset points', fontsize=7)
    corr = queue_results.get('cg_queue_correlations', {})
    pearson, spearman = corr.get('pearson'), corr.get('spearman')
    annotation = (
        f"Pearson r = {pearson:.2f}\nSpearman ρ = {spearman:.2f}"
        if pearson is not None and spearman is not None else 'Correlation undefined'
    )
    ax2.text(0.03, 0.97, annotation, transform=ax2.transAxes, va='top',
             bbox={'facecolor': 'white', 'edgecolor': COLORS['light'], 'pad': 4})
    ax2.set_xlabel('Normalized CG objective')
    ax2.set_ylabel('Normalized queue objective')
    ax2.set_title('Cross-model placement agreement')
    clean_axis(ax2)
    save_publication_figure(fig, output_path)
    plt.close(fig)
    print(f"Objective comparison plot saved to {output_path}")


def _plot_placement_search_comparison(cg_search, queue_search, output_path):
    """Show evaluation order and final algorithm choices for both models."""
    if not cg_search or not queue_search:
        return
    phase_style = {
        'greedy': (COLORS['blue'], 'o'),
        'single_swap': (COLORS['green'], 's'),
        'exhaustive': (COLORS['orange'], '^'),
    }
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5))
    model_results = {}
    for ax, title, search, ylabel in (
        (axes[0], 'Congestion-game placement search', cg_search, 'CG objective'),
        (axes[1], 'Queue placement search (paired replication mean)', queue_search,
         'Queue total travel time'),
    ):
        trace = search.get('trace', [])
        x = np.arange(1, len(trace) + 1)
        y = np.asarray([item['objective'] for item in trace], dtype=float)
        ax.plot(x, y, color=COLORS['light'], linewidth=0.9, alpha=0.9, zorder=1)
        for phase, (color, marker) in phase_style.items():
            indices = [i for i, item in enumerate(trace) if item['phase'] == phase]
            if indices:
                ax.scatter(
                    x[indices], y[indices], color=color, marker=marker, s=55,
                    zorder=2,
                )
        placement_to_point = {
            tuple(item['placement']): (index + 1, item['objective'])
            for index, item in enumerate(trace)
        }
        for method in ('greedy', 'single_swap', 'exhaustive'):
            choice = search.get(method, {})
            point = placement_to_point.get(tuple(choice.get('placement', [])))
            if point:
                ax.scatter(
                    [point[0]], [point[1]], facecolors='none', edgecolors=COLORS['red'],
                    linewidths=1.8, s=125, zorder=3,
                )
            if choice:
                model_results.setdefault(title.split(' placement')[0], []).append(
                    f"{method.replace('_', ' ').title()} "
                    f"{'+'.join(map(str, choice['placement']))} ({choice['objective']:.3g})"
                )
        for index in range(1, len(trace)):
            if trace[index]['phase'] != trace[index - 1]['phase']:
                ax.axvline(index + 0.5, color=COLORS['light'], linewidth=0.8,
                           linestyle='--', zorder=0)
        labels = ['+'.join(map(str, item['placement'])) for item in trace]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=50, ha='right', fontsize=8)
        ax.set_xlabel('Unique placement evaluation order')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        clean_axis(ax)
    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=COLORS['blue'], markeredgecolor=COLORS['blue'],
               markersize=8, label='Greedy'),
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=COLORS['green'], markeredgecolor=COLORS['green'],
               markersize=8, label='Single swap'),
        Line2D([0], [0], marker='^', color='w',
               markerfacecolor=COLORS['orange'], markeredgecolor=COLORS['orange'],
               markersize=8, label='Exhaustive'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=COLORS['red'], markeredgewidth=1.8,
               markersize=10, label='Chosen placement'),
    ]
    fig.legend(
        handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, 0.10),
        ncol=4, frameon=True, facecolor='white', edgecolor=COLORS['light'],
    )
    result_lines = [
        f"{model}:  " + "   |   ".join(values)
        for model, values in model_results.items()
    ]
    if result_lines:
        fig.text(0.5, 0.02, "\n".join(result_lines), ha='center', va='top',
                 fontsize=8, color=COLORS['dark'])
    fig.subplots_adjust(bottom=0.24, top=0.93, hspace=0.55, right=0.98)
    save_publication_figure(fig, output_path)
    plt.close(fig)
    print(f"Placement search comparison saved to {output_path}")


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
        phases = list(network_stages.keys())
        nodes = [network_stages[p].get('nodes', 0) for p in phases]
        edges = [network_stages[p].get('edges', 0) for p in phases]
    else:
        n = node_count or 0
        e = edge_count or 0
        phases = ['original (no cleaning)']
        nodes = [n]
        edges = [e]

    if has_maps:
        map_phases = list(stage_maps.keys())
        n_maps = len(map_phases)
        if n_maps == 1:
            ncols, map_rows = 1, 1
            fig = plt.figure(figsize=(10.5, 4.5), constrained_layout=True)
            grid = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0])
            ax_bar = fig.add_subplot(grid[0, 0])
        else:
            ncols = min(n_maps, 3)
            map_rows = (n_maps + ncols - 1) // ncols
            fig = plt.figure(
                figsize=(3.6 * ncols, 3.0 + 3.2 * map_rows),
                constrained_layout=True,
            )
            grid = fig.add_gridspec(
                1 + map_rows, ncols,
                height_ratios=[1.0] + [1.25] * map_rows,
            )
            ax_bar = fig.add_subplot(grid[0, :])
    else:
        fig, ax_bar = plt.subplots(figsize=(10, 5))

    # Bar chart (always)
    x = np.arange(len(phases))
    w = 0.35
    node_bars = ax_bar.bar(x - w/2, nodes, w, color=COLORS['blue'], label='Nodes')
    edge_bars = ax_bar.bar(x + w/2, edges, w, color=COLORS['orange'], label='Directed edges')
    ax_bar.set_ylabel('Count')
    ax_bar.set_title('Network Cleaning Pipeline — Nodes & Edges per Phase')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(phases, rotation=30, ha='right', fontsize=8)
    ax_bar.legend()
    clean_axis(ax_bar)
    for bar, val in zip(list(node_bars) + list(edge_bars), nodes + edges):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(edges)*0.01,
                    str(val), ha='center', va='bottom', fontsize=7)

    # Map grid (when available)
    if has_maps:
        for idx, phase in enumerate(map_phases):
            ax = (
                fig.add_subplot(grid[0, 1])
                if n_maps == 1
                else fig.add_subplot(grid[1 + idx // ncols, idx % ncols])
            )
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
                                linewidth=1.15, color=COLORS['mid'], alpha=0.78, zorder=1)
            else:
                for u, v in mp['edges_pairs']:
                    if u in pos_by_id and v in pos_by_id:
                        ax.plot([pos_by_id[u][0], pos_by_id[v][0]],
                                [pos_by_id[u][1], pos_by_id[v][1]],
                                linewidth=1.15, color=COLORS['mid'], alpha=0.78, zorder=1)
            marker_size = max(10.0, min(28.0, 2200.0 / max(1, len(xs))))
            ax.scatter(xs, ys, s=marker_size, c=COLORS['blue'], alpha=0.9,
                       edgecolors='white', linewidths=0.5, zorder=2)
            ax.set_title(f"{phase}\nnodes={mp['_n']} edges={mp['_e']}",
                         fontsize=7)
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xticks([])
            ax.set_yticks([])

        if n_maps > 1:
            for idx in range(n_maps, ncols * map_rows):
                fig.add_subplot(grid[1 + idx // ncols, idx % ncols]).set_visible(False)

    if not has_maps:
        fig.tight_layout()
    save_publication_figure(fig, output_path)
    plt.close(fig)


def _plot_bpr_fit_samples(pandas_df, output_path, n_random=6, n_worst=6, seed=0):
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
    n = min(len(sample), 12)

    rows = 3; cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(11, 7.8))
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
                        capacity = float(row.get('cap_fit', 1.0))
                        normalized_x = np.asarray(xv, dtype=float) / capacity
                        ax.scatter(normalized_x, yv, s=15, color=COLORS['blue'], alpha=0.8)
                        zero = np.isclose(normalized_x, 0.0)
                        if zero.any():
                            ax.scatter(
                                normalized_x[zero], np.asarray(yv)[zero], s=18,
                                color=COLORS['orange'], marker='D', zorder=3,
                            )
                        a_fit = row.get('a_fit', np.nan)
                        fft_fit = row.get('fft_fit', np.nan)
                        if np.isfinite(a_fit) and np.isfinite(fft_fit):
                            if float(a_fit) > 0:
                                xs = np.linspace(min(normalized_x), max(normalized_x), 100)
                                a, b, f = a_fit, row['b_fit'], fft_fit
                                ys = f * (1 + a * xs**b)
                                ax.plot(xs, ys, color=COLORS['red'], linewidth=1.3)
                            else:
                                ax.axhline(
                                    y=float(fft_fit), color=COLORS['orange'],
                                    linewidth=1, linestyle='--',
                                    label='Constant fit',
                                )
                except Exception:
                    pass
            r2 = row.get('R^2', np.nan)
            status = row.get('fit_status', 'unknown')
            ax.set_title(
                f'Link {int(row["link_id"])} {status} R²={r2:.3f}',
                fontsize=7,
            )
            ax.axvline(1.0, color='0.6', linestyle=':', linewidth=0.6)
            ax.set_xticks([0, 1, 2])
            ax.tick_params(axis='both', labelsize=7)
            clean_axis(ax)
        else:
            ax.set_xticks([]); ax.set_yticks([])
    for i in range(n, rows*cols):
        axes[i].set_visible(False)
    fig.suptitle('BPR fit diagnostics: random and lowest-R² links', fontsize=12)
    fig.supxlabel('Offered cohort / capacity (orange diamond = zero-flow probe)', fontsize=9)
    fig.supylabel('Entry-wait-inclusive travel time (s)', fontsize=9)
    fig.tight_layout()
    save_publication_figure(fig, output_path)
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
    axes[0].bar(labels, [len(fresh)], color=COLORS['blue'], label='fresh')
    if reference.size:
        axes[0].bar(['historical reference'], [len(reference)], color=COLORS['orange'], label='reference')
    axes[0].set_ylabel('links with fitted R²')
    axes[0].set_title('BPR coverage')
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].legend(fontsize=8)
    clean_axis(axes[0])

    bins = np.linspace(0, 1.01, 21)
    if reference.size:
        axes[1].hist(reference, bins=bins, alpha=0.55, color=COLORS['orange'], label=f'reference {reference_commit}')
    if fresh.size:
        axes[1].hist(fresh, bins=bins, alpha=0.55, color=COLORS['blue'], label='fresh current network')
    axes[1].set_xlabel('R² (legacy-compatible field)')
    axes[1].set_ylabel('link count')
    axes[1].set_title('Historical-compatible fit quality')
    axes[1].legend(fontsize=8)
    clean_axis(axes[1])
    fresh_status = pandas_df.get('fit_status', pd.Series(dtype=str)).value_counts().to_dict()
    fig.suptitle(
        'Historical BPR compatibility — status comparison\n'
        f'fresh={fresh_status}; reference={reference_status or "unavailable"}',
        fontsize=10,
    )
    fig.tight_layout()
    save_publication_figure(fig, output_path)
    plt.close(fig)


def _plot_ne_convergence(convergence_data, output_path, queue_manifest=None):
    """Plot NE convergence curves: diff vs iteration per config."""
    if not convergence_data:
        return
    statuses = (queue_manifest or {}).get('configuration_statuses', {})
    status_colors = {
        'converged': COLORS['green'], 'cycle': COLORS['orange'],
        'nonconverged': COLORS['red'], 'failed': COLORS['red'],
    }
    fig, (ax, ax_hist) = plt.subplots(1, 2, figsize=(10.5, 4.1), constrained_layout=True)
    shown = set()
    for config_str, diffs in convergence_data.items():
        if diffs:
            status = statuses.get(config_str, 'nonconverged')
            label = status.title() if status not in shown else None
            shown.add(status)
            ax.plot(range(1, len(diffs) + 1), diffs, linewidth=1.1, alpha=0.72,
                    marker='o', markersize=2.8,
                    color=status_colors.get(status, COLORS['mid']), label=label)
    alpha = float((queue_manifest or {}).get('alpha', 0.01))
    ax.axhline(y=alpha, color=COLORS['dark'], linestyle='--', linewidth=0.9,
               label=f'Tolerance ({alpha:g})')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Relative route travel-time gap')
    ax.set_title('Better-response trajectories')
    ax.legend(loc='upper right')
    maximum_iteration = max((len(value) for value in convergence_data.values()), default=1)
    ax.set_xlim(0.5, maximum_iteration + 0.5)
    if maximum_iteration <= 12:
        ax.set_xticks(range(1, maximum_iteration + 1))
    finite_gaps = [
        float(value) for values in convergence_data.values() for value in values
        if np.isfinite(value)
    ]
    ax.set_ylim(0, max(alpha * 1.5, max(finite_gaps, default=0.0) * 1.08, 1e-4))
    clean_axis(ax)
    iterations = [len(value) for value in convergence_data.values()]
    bins = np.arange(0.5, max(iterations, default=1) + 1.5, 1)
    ax_hist.hist(iterations, bins=bins, color=COLORS['blue'], alpha=0.85,
                 edgecolor='white')
    ax_hist.set_xlabel('Iterations completed')
    ax_hist.set_ylabel('Configurations')
    ax_hist.set_title('Iteration-count distribution')
    if max(iterations, default=1) <= 12:
        ax_hist.set_xticks(range(1, max(iterations, default=1) + 1))
    clean_axis(ax_hist)
    save_publication_figure(fig, output_path)
    plt.close(fig)


def _plot_timing_breakdown(timing, output_path):
    """Plot timing breakdown as horizontal bar chart."""
    steps = {k: v for k, v in timing.items() if k != 'total' and v > 0}
    if not steps:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(steps.keys())
    values = list(steps.values())
    palette = [COLORS['blue'], COLORS['green'], COLORS['orange'], COLORS['purple'], COLORS['red']]
    colors = [palette[index % len(palette)] for index in range(len(names))]
    bars = ax.barh(names, values, color=colors)
    total = sum(values)
    for bar, val in zip(bars, values):
        pct = val / total * 100 if total > 0 else 0
        ax.text(bar.get_width() + max(values)*0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.0f}s ({pct:.0f}%)', va='center', fontsize=8)
    ax.set_xlabel('Wall-clock time (s)')
    ax.set_title(f'Timing Breakdown (total: {total:.0f}s)')
    clean_axis(ax, grid_axis='x')
    fig.tight_layout()
    save_publication_figure(fig, output_path)
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
        cg_search = cg_results.get('placement_search', {})
        for method in ('greedy', 'single_swap', 'exhaustive'):
            outcome = cg_search.get(method)
            if outcome:
                lines.append(
                    f"  {method.replace('_', ' ').title():<12s}: "
                    f"{outcome['placement']}  objective={outcome['objective']:.4f}"
                )
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
        ne_stats = queue_results.get('ne_statistics', {})
        status_counts = ne_stats.get('status_counts', {})
        iteration_stats = ne_stats.get('iteration_statistics', {}).get('all', {})
        greedy_label = (
            'Greedy + single swap' if qc.get('single_swap') else 'Greedy'
        )
        lines.append(
            f"  Assignment quality: {queue_results.get('assignment_quality', 'unknown')}"
        )
        if queue_results.get('uses_approximate_ne'):
            lines.append(
                "  WARNING: cycle states were retained as approximate assignments; "
                "they are not verified Nash equilibria."
            )
        if status_counts:
            lines.append(
                "  NE statuses: "
                + ", ".join(
                    f"{name}={int(status_counts.get(name, 0))}"
                    for name in ('converged', 'cycle', 'nonconverged', 'failed')
                )
            )
        if iteration_stats.get('count'):
            lines.append(
                "  NE iterations: "
                f"min={iteration_stats['min']:.0f}, "
                f"median={iteration_stats['median']:.1f}, "
                f"mean={iteration_stats['mean']:.1f}, "
                f"p95={iteration_stats['p95']:.1f}, "
                f"max={iteration_stats['max']:.0f}"
            )
        lines.append(f"  K-routes:   {qc.get('K', 'N/A')}")
        lines.append(f"  MC reps:    {qc.get('N', 'N/A')}")
        lines.append(f"  Single-swap: {qc.get('single_swap', 'N/A')}")
        lines.append(f"  {greedy_label} best: {queue_results['best_greedy']['positions']}  "
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
        correlations = queue_results.get('cg_queue_correlations', {})
        if correlations.get('paired_count', correlations.get('n', 0)):
            lines.append(
                "  CG-Queue correlations: "
                f"Pearson={correlations.get('pearson')}, "
                f"Spearman={correlations.get('spearman')}, "
                f"paired placements={correlations.get('paired_count', correlations.get('n'))}"
            )

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

    bpr_manifest_path = os.path.join(experiment_dir, 'bpr', 'bpr_manifest.json')
    bpr_results_path = os.path.join(experiment_dir, 'bpr', 'fitter_results.csv')
    if os.path.isfile(bpr_manifest_path):
        with open(bpr_manifest_path) as handle:
            bpr_manifest = json.load(handle)
        lines.extend([
            "",
            "## BPR Calibration",
            "",
            f"- Method version: `{bpr_manifest.get('fitter_version', 'unknown')}`",
            f"- Route measurement: `{bpr_manifest.get('route_semantics', 'unknown')}`",
            f"- Flow unit: `{bpr_manifest.get('flow_unit', 'unknown')}`",
            f"- Calibration window: {bpr_manifest.get('calibration_window_hours', 'N/A')} hours",
            f"- Flow fractions: {bpr_manifest.get('flow_fractions', 'N/A')}",
            f"- Fit statuses: {bpr_manifest.get('fit_status_counts', {})}",
        ])
        if os.path.isfile(bpr_results_path):
            bpr_results = pd.read_csv(bpr_results_path)
            r2 = pd.to_numeric(
                bpr_results.get('R^2', pd.Series(dtype=float)), errors='coerce'
            ).dropna()
            rmse = pd.to_numeric(
                bpr_results.get('fit_rmse_seconds', pd.Series(dtype=float)),
                errors='coerce'
            ).dropna()
            median_relative = pd.to_numeric(
                bpr_results.get(
                    'fit_median_relative_error', pd.Series(dtype=float)
                ), errors='coerce'
            ).dropna()
            if not r2.empty:
                lines.append(
                    f"- R² min / median: {r2.min():.3f} / {r2.median():.3f}"
                )
            if not rmse.empty:
                lines.append(f"- Median link RMSE: {rmse.median():.2f} seconds")
            if not median_relative.empty:
                lines.append(
                    "- Median of per-link median relative errors: "
                    f"{100 * median_relative.median():.1f}%"
                )

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
        cg_search = cg_results.get('placement_search', {})
        lines.extend([
            "",
            "## Congestion-Game Results",
            "",
            f"- Configurations evaluated: {cg_results.get('num_configs', 'N/A')}",
            f"- Best placement: {cg_results.get('best_chargers', 'N/A')}",
            f"- Best objective (total delay): {cg_results.get('best_objective', 'N/A'):.4f}" if isinstance(cg_results.get('best_objective'), (int, float)) else f"- Best objective: {cg_results.get('best_objective', 'N/A')}",
        ])
        if cg_search:
            lines.extend([
                "",
                "### CG Algorithm Outcomes",
                "",
                "| Method | Placement | Objective | Phase wall time (s) | Worker work (s) |",
                "|---|---|---:|---:|---:|",
            ])
            cg_wall = cg_search.get('phase_wall_seconds', {})
            cg_work = cg_search.get('phase_worker_seconds', {})
            for method in ('greedy', 'single_swap', 'exhaustive'):
                outcome = cg_search.get(method, {})
                if not outcome:
                    continue
                lines.append(
                    f"| {method.replace('_', ' ').title()} | {outcome['placement']} | "
                    f"{outcome['objective']:.4f} | {cg_wall.get(method, 0.0):.2f} | "
                    f"{cg_work.get(method, 0.0):.2f} |"
                )
        lines.extend([
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
        greedy_label = (
            "Greedy + single swap"
            if queue_results.get('config', {}).get('single_swap') else "Greedy"
        )
        lines.extend([
            "",
            "## Queue-Based Simulation Results",
            "",
            f"**Assignment quality:** `{queue_results.get('assignment_quality', 'unknown')}`",
            "",
        ])
        if queue_results.get('uses_approximate_ne'):
            lines.extend([
                "**Important limitation:** Better-response dynamics cycled. The "
                "comparison below uses the current assignment retained at cycle "
                "detection. These assignments are approximations and have not "
                "been verified as Nash equilibria.",
                "",
            ])
        ne_stats = queue_results.get('ne_statistics', {})
        status_counts = ne_stats.get('status_counts', {})
        iteration_stats = ne_stats.get('iteration_statistics', {})
        if status_counts:
            total_ne = sum(int(value) for value in status_counts.values())
            lines.extend([
                "### NE termination statistics",
                "",
                "| Status | Configurations | Fraction | Median iterations | Mean | p95 | Maximum |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ])
            for status in ('converged', 'cycle', 'nonconverged', 'failed'):
                count = int(status_counts.get(status, 0))
                stats = iteration_stats.get(status, {})
                fraction = count / total_ne if total_ne else 0.0
                def _stat(name):
                    value = stats.get(name)
                    return "—" if value is None else f"{value:.1f}"
                lines.append(
                    f"| {status.replace('_', ' ').title()} | {count} | {fraction:.1%} | "
                    f"{_stat('median')} | {_stat('mean')} | {_stat('p95')} | {_stat('max')} |"
                )
            all_stats = iteration_stats.get('all', {})
            if all_stats.get('count'):
                lines.append(
                    f"| **All** | **{int(all_stats['count'])}** | **100.0%** | "
                    f"**{all_stats['median']:.1f}** | **{all_stats['mean']:.1f}** | "
                    f"**{all_stats['p95']:.1f}** | **{all_stats['max']:.1f}** |"
                )
            cycle_stats = ne_stats.get('cycle_length_statistics', {})
            if cycle_stats.get('count'):
                lines.extend([
                    "",
                    "For detected cycles, cycle length had "
                    f"median `{cycle_stats['median']:.1f}`, p95 `{cycle_stats['p95']:.1f}`, "
                    f"and maximum `{cycle_stats['max']:.1f}` iterations.",
                ])
            lines.append("")
        if queue_results.get('timing', {}).get('paired_placement_cache'):
            lines.extend([
                "Greedy and exhaustive use the same canonical placement identity "
                "and paired Monte Carlo realization. A placement appearing in both "
                "tables was simulated only once per replication and reused.",
                "",
            ])
        queue_search = queue_results.get('placement_search', {})
        if queue_search:
            lines.extend([
                "### Queue Algorithm Outcomes",
                "",
                "| Method | Placement | Avg travel time | Incremental worker work (s) |",
                "|---|---|---:|---:|",
            ])
            queue_work = queue_search.get('phase_worker_seconds', {})
            for method in ('greedy', 'single_swap', 'exhaustive'):
                outcome = queue_search.get(method, {})
                if not outcome:
                    continue
                lines.append(
                    f"| {method.replace('_', ' ').title()} | {outcome['placement']} | "
                    f"{outcome['objective']:.1f} | {queue_work.get(method, 0.0):.2f} |"
                )
            lines.append("")
        lines.extend([
            "### All Exhaustive Queue Placements",
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
            f"- Best {greedy_label.lower()}: {queue_results['best_greedy']['positions']} (avg TT = {queue_results['best_greedy']['avg_travel_time']:.1f})",
            f"- Best exhaustive: {queue_results['best_exhaustive']['positions']} (avg TT = {queue_results['best_exhaustive']['avg_travel_time']:.1f})",
            f"- Greedy suboptimality: {queue_results['suboptimality_pct']:.2f}%",
            f"- Monte Carlo reps: {queue_results['config']['N']}",
            f"- K (routes): {queue_results['config']['K']}",
            f"- Single swap: {queue_results['config']['single_swap']}",
        ])
        if queue_results.get('timing', {}).get('paired_placement_cache'):
            lines.extend([
                f"- Unique queue simulations: {queue_results['timing']['unique_simulations']}",
                f"- Avoided duplicate simulations: {queue_results['timing']['placement_cache_hits']}",
            ])
    elif not QUEUE_SIM_AVAILABLE:
        lines.extend([
            "",
            "## Queue-Based Simulation",
            "",
            f"**SKIPPED** — a platform-native liblsp shared library is unavailable.",
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
            f"| Congestion game | {cg_best} | {cg_results.get('best_objective', 'N/A'):.4f} |" if isinstance(cg_results.get('best_objective'), (int, float)) else f"| Congestion game | {cg_best} | N/A |",
            f"| Queue {greedy_label.lower()} | {q_greedy} | {queue_results['best_greedy']['avg_travel_time']:.1f} |",
            f"| Queue exhaustive | {q_exhaustive} | {queue_results['best_exhaustive']['avg_travel_time']:.1f} |",
            "",
        ])
        cg_best_set = set(int(x) for x in cg_best) if cg_best is not None else set()
        q_best_set = set(q_exhaustive)
        match = "YES" if cg_best_set == q_best_set else "NO"
        lines.append(f"CG and queue agree on optimal placement: **{match}**")
        correlations = queue_results.get('cg_queue_correlations', {})
        paired_count = correlations.get('paired_count', correlations.get('n', 0))
        if paired_count:
            pearson = correlations.get('pearson')
            spearman = correlations.get('spearman')
            pearson_text = "undefined" if pearson is None else f"{pearson:.3f}"
            spearman_text = "undefined" if spearman is None else f"{spearman:.3f}"
            lines.extend([
                "",
                "### Placement-objective association",
                "",
                f"Across `{paired_count}` identical final-size placements:",
                "",
                f"- Pearson correlation: `{pearson_text}`",
                f"- Spearman rank correlation: `{spearman_text}`",
                "",
                "These correlations compare model rankings over the same charger sets; "
                "they do not establish equality of the differently scaled objectives.",
            ])
            if queue_results.get('uses_approximate_ne'):
                lines.extend([
                    "Because at least one queue assignment is a retained cycle state, "
                    "the correlations are descriptive of the current approximations, not "
                    "correlations between verified Nash-equilibrium outcomes.",
                ])

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
        f"| `plots/generated_scenario_comparison.png` | Scenario: plain vs OSM HOT vs Positron panels (only when tile fetch succeeds) |",
        f"| `plots/ne_convergence.png` | NE convergence: diff vs iteration per config |",
        f"| `plots/timing_breakdown.png` | Pipeline step durations |",
        f"| `plots/bpr_fit_samples.png` | BPR fit diagnostics (random + worst-R² links) |",
        f"| `plots/objective_comparison.png` | Paired CG–Queue objectives and Pearson/Spearman association |",
        f"| `plots/placement_search_comparison.png` | Ordered greedy, one-swap, and exhaustive search outcomes |",
        f"| `placement_search_summary.json` | Algorithm choices, candidate order, and phase timings |",
        f"| `placement_search_trace.csv` | Machine-readable placement evaluation sequence |",
    ])
    if queue_results:
        lines.extend([
            f"| `queue/NE_path_assignments.pkl` | Queue route counts with exact/approximate status per config |",
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
    bpr_config = dict(config.pipeline.get("bpr_generation", {}))
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
    capacity_per_lane = float(bpr_config.get('capacity_per_lane', 1900.0))
    shared_road_net.edges['capacity'] = (
        pd.to_numeric(shared_road_net.edges['lanes'], errors='raise')
        * capacity_per_lane
    )
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
            'capacity_per_directional_lane_vph': capacity_per_lane,
            'random_seed': seed_manager.seed,
        },
    )
    with open(os.path.join(experiment_dir, 'network_manifest.json'), 'w') as handle:
        json.dump(network_manifest, handle, indent=2, default=str)

    scenario_metadata = None
    if config.scenario_generation.get("enabled", False):
        from src.scenario_generation import generate_scenario, plot_scenario, plot_scenario_comparison
        generated = generate_scenario(
            shared_road_net,
            config.scenario_generation,
            calibration_window_hours=float(
                bpr_config.get('calibration_window_hours', 0.1)
            ),
        )
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
        # 3-panel plain/HOT/Positron figure; skipped when tiles unavailable.
        plot_scenario_comparison(
            shared_road_net, generated,
            os.path.join(plot_dir, "generated_scenario_comparison.png"),
        )
    atomic_write_json(os.path.join(experiment_dir, "status.json"), {
        "status": "running", "stage": "bpr", "config_digest": digest,
        "network_hash": network_manifest["network_hash"], **process_provenance(),
    })

    # Step 1: BPR fitting
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

    bpr_fit_plot = os.path.join(plot_dir, 'bpr_fit_samples.png')
    try:
        _plot_bpr_fit_samples(pandas_df, bpr_fit_plot, seed=seed_manager.seed)
        if not os.path.exists(bpr_fit_plot):
            raise RuntimeError('plot function did not create an output file')
        print(f"BPR fit diagnostics saved to {bpr_fit_plot}")
    except Exception as exc:
        raise RuntimeError(f'Failed to create required BPR fit diagnostics: {exc}') from exc

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
        target_grids = [
            grid for grid in grids
            if len(set(int(value) for value in grid.chargers)) == config.num_chargers
        ]
        if not target_grids:
            raise RuntimeError('CG optimization produced no target-size placement')
        best_grid = min(target_grids, key=lambda grid: grid.travel_time_obj)
        all_opt_path = os.path.join(experiment_dir, 'all_optimization_results.pkl')
        with open(all_opt_path, 'rb') as handle:
            optimization_data = pickle.load(handle)
        cg_search = optimization_data.get('placement_search', {})
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
            'placement_search': cg_search,
        }
        grid_solver_metadata = getattr(
            best_grid, 'solver_metadata', getattr(best_grid.net, 'solver_metadata', {})
        )
        solver_metadata.update(grid_solver_metadata)
        if os.path.exists(all_opt_path):
            # Attach the exact artifact identity and solver provenance to the
            # object consumed by route recovery and queue stages.
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
            resume=resume,
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
        if queue_manifest.get('approximate_configurations'):
            print(
                'WARNING: cycle detection retained the current assignment as '
                'an approximate equilibrium for '
                f'{len(queue_manifest["approximate_configurations"])} '
                'configurations. Downstream results will be labeled approximate.'
            )

        if convergence_data:
            _save_convergence_csv(convergence_data, os.path.join(experiment_dir, 'queue', 'ne_convergence.csv'))
            print(f"Convergence data saved to {experiment_dir}/queue/ne_convergence.csv")
            _plot_ne_convergence(
                convergence_data,
                os.path.join(plot_dir, 'ne_convergence.png'),
                queue_manifest,
            )

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
            _plot_placement_search_comparison(
                cg_results.get('placement_search'),
                queue_results.get('placement_search'),
                os.path.join(plot_dir, 'placement_search_comparison.png'),
            )
            if (cg_results.get('placement_search')
                    and queue_results.get('placement_search')):
                search_summary = {
                    'candidate_order': [
                        int(value) for value in config.possible_charger_positions
                    ],
                    'congestion_game': cg_results['placement_search'],
                    'queue': queue_results['placement_search'],
                }
                atomic_write_json(
                    os.path.join(experiment_dir, 'placement_search_summary.json'),
                    search_summary,
                )
                trace_rows = []
                for model, search in (
                    ('congestion_game', cg_results['placement_search']),
                    ('queue', queue_results['placement_search']),
                ):
                    for item in search.get('trace', []):
                        trace_rows.append({'model': model, **item})
                pd.DataFrame(trace_rows).to_csv(
                    os.path.join(experiment_dir, 'placement_search_trace.csv'),
                    index=False,
                )
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
        **process_provenance(),
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

    # A full recursive inventory can issue hundreds of thousands of metadata
    # operations for raw queue/BPR timestep files. On shared Slurm storage it
    # can outlast the experiment while consuming almost no CPU. Keep it
    # opt-in and write a small, explicit marker by default.
    full_inventory = bool(config.pipeline.get('full_artifact_inventory', False))
    if full_inventory:
        inventory = directory_inventory(experiment_dir)
        inventory.update({
            'full_scan': True,
            'scan_status': 'complete',
        })
    else:
        inventory = {
            'full_scan': False,
            'scan_status': 'skipped',
            'reason': (
                'Recursive artifact inventory disabled to avoid expensive '
                'shared-filesystem metadata traversal'
            ),
            'file_count': None,
            'total_bytes': None,
            'files': [],
        }
    atomic_write_json(os.path.join(experiment_dir, "artifact_inventory.json"), inventory)
    bpr_manifest_path = os.path.join(experiment_dir, 'bpr', 'bpr_manifest.json')
    bpr_manifest = {}
    if os.path.isfile(bpr_manifest_path):
        with open(bpr_manifest_path) as handle:
            bpr_manifest = json.load(handle)
    uses_approximate_ne = bool(
        queue_results and queue_results.get('uses_approximate_ne', False)
    )
    completion_status = (
        'complete_with_approximate_ne' if uses_approximate_ne else 'complete'
    )
    strict_eligible = not uses_approximate_ne
    summary_row = {
        'run_id': os.path.basename(experiment_dir),
        'status': completion_status,
        'eligible': strict_eligible,
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
        "status": completion_status, "stage": "complete",
        "eligible": strict_eligible,
        "uses_approximate_ne": uses_approximate_ne,
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
    try:
        if args.validate_config:
            validated = Config.from_json(args.config)
            print(json.dumps(validated.to_dict(), indent=2, sort_keys=True))
        elif args.network_only and args.pruning_sweep:
            parser.error('--network-only and --pruning-sweep are mutually exclusive')
        elif args.pruning_sweep:
            from src.pruning_study import run_pruning_sweep
            run_pruning_sweep(args.config)
        elif args.network_only:
            run_network_only(args.config)
        else:
            run_pipeline(args.config, results_root=args.results_root, resume=args.resume)
    finally:
        _cleanup_multiprocessing_children()
