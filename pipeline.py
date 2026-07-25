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
import json
import os
import sys
import time
import pickle
import platform
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.model_fitter import TrafficModelFitter, convert_string_to_array
from src.utils import outer_optimization

try:
    from queue_sim import QUEUE_SIM_AVAILABLE, _QUEUE_SIM_ERROR
except (ImportError, ModuleNotFoundError):
    QUEUE_SIM_AVAILABLE = False
    _QUEUE_SIM_ERROR = "queue_sim package not found (macOS-only)"


def _fill_missing_links_to_count(pandas_df, target_count):
    """Ensure the BPR fit DataFrame covers all link_ids 0..target_count-1."""
    import pandas as pd
    import numpy as np
    existing = set(pandas_df['link_id'].unique())
    missing = sorted(set(range(target_count)) - existing)
    if not missing:
        return pandas_df
    n_samples = len(pandas_df.iloc[0]['x_vector']) if 'x_vector' in pandas_df.columns else 25
    rows = []
    for lid in missing:
        rows.append({
            'link_id': lid,
            'x_vector': np.zeros(n_samples),
            'y_vector': np.zeros(n_samples),
            'a_fit': 0.0, 'b_fit': 0.0, 'cap_fit': 1.0, 'fft_fit': 1.0, 'R^2': 1.0,
        })
    df = pd.concat([pandas_df, pd.DataFrame(rows)], ignore_index=True).sort_values('link_id').reset_index(drop=True)
    print(f"Filled {len(missing)} missing link_ids (up to {target_count})")
    return df


def load_or_fit_model(data_path="data/traffic_data.csv", cache_path="data/cached_results.pkl",
                      coordinates=None, bpr_config=None, work_dir=None, n_links=None,
                      road_filter_config=None):
    """Load cached BPR fit, fit from existing data, or generate data + fit.

    Priority:
    1. If cache exists → load it
    2. If traffic_data.csv exists → fit from it
    3. If coordinates provided → generate data via queue sim, then fit

    Args:
        n_links: If provided, ensure the fit covers link_ids 0..n_links-1.
        road_filter_config: Dict with 'highway_types' and 'prune_dead_ends' keys.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            pandas_df, model_fitter = pickle.load(f)
        if n_links and len(pandas_df) < n_links:
            pandas_df = _fill_missing_links_to_count(pandas_df, n_links)
        print("Loaded cached BPR fit results.")
    elif os.path.exists(data_path):
        print("No cache found. Fitting BPR models from existing data...")
        import pandas as pd
        pandas_df = pd.read_csv(data_path)
        convert_string_to_array(pandas_df, 'x_vector')
        convert_string_to_array(pandas_df, 'y_vector')
        model_fitter = TrafficModelFitter(pandas_df=pandas_df)
        model_fitter.parallel_fit_and_evaluate()
        model_fitter.fill_missing_link_ids()
        pandas_df = model_fitter.df
        if n_links and len(pandas_df) < n_links:
            pandas_df = _fill_missing_links_to_count(pandas_df, n_links)
        with open(cache_path, "wb") as f:
            pickle.dump((pandas_df, model_fitter), f)
        print("Cached BPR fit results.")
    elif coordinates is not None and QUEUE_SIM_AVAILABLE:
        print("No cache or data found. Generating BPR data via queue simulation...")
        from queue_sim.bpr_data_generator import generate_and_save_bpr_data
        num_samples = (bpr_config or {}).get('num_samples', 25)
        max_flow = (bpr_config or {}).get('max_flow', 250)
        rf = road_filter_config or {}
        highway_types = rf.get('highway_types') if rf.get('enabled', True) else None
        prune_de = rf.get('prune_dead_ends', True)
        _, n_links_generated = generate_and_save_bpr_data(
            coordinates, data_path,
            num_samples=num_samples, max_flow=max_flow, work_dir=work_dir,
            highway_types=highway_types, prune_dead_ends=prune_de,
        )
        import pandas as pd
        pandas_df = pd.read_csv(data_path)
        convert_string_to_array(pandas_df, 'x_vector')
        convert_string_to_array(pandas_df, 'y_vector')
        model_fitter = TrafficModelFitter(pandas_df=pandas_df)
        model_fitter.parallel_fit_and_evaluate()
        model_fitter.fill_missing_link_ids()
        pandas_df = model_fitter.df
        if n_links and len(pandas_df) < n_links:
            pandas_df = _fill_missing_links_to_count(pandas_df, n_links)
        elif n_links_generated and len(pandas_df) < n_links_generated:
            pandas_df = _fill_missing_links_to_count(pandas_df, n_links_generated)
        with open(cache_path, "wb") as f:
            pickle.dump((pandas_df, model_fitter), f)
        print("Generated BPR data and cached fit results.")
    else:
        raise FileNotFoundError(
            f"No BPR data found at {data_path} or {cache_path}. "
            f"Provide coordinates and run on macOS to generate data via queue simulation."
        )
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

    all_labels = sorted(set(cg_labels) | set(q_labels))
    cg_map = dict(zip(cg_labels, cg_norm))
    q_map = {str(r['positions']): r['avg_travel_time'] / q_min for r in q_results if r['avg_travel_time'] != float('inf')}

    cg_bars = [cg_map.get(l, 0) for l in all_labels]
    q_bars = [q_map.get(l, 0) for l in all_labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: normalized comparison
    x = np.arange(len(all_labels))
    width = 0.35
    ax1.bar(x - width/2, cg_bars, width, label='Congestion Game', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, q_bars, width, label='Queue Simulation', color='coral', alpha=0.8)
    ax1.set_ylabel('Normalized objective (best = 1.0)')
    ax1.set_title('CG vs Queue: Objective per Placement (Normalized)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(all_labels, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Right: raw values (twin axis for different scales)
    ax2.bar(x - width/2, cg_vals, width, label='CG objective', color='steelblue', alpha=0.8)
    ax2.set_ylabel('CG total delay', color='steelblue')
    ax2.tick_params(axis='y', labelcolor='steelblue')
    ax2.set_xticks(x)
    ax2.set_xticklabels(all_labels, rotation=45, ha='right')
    ax2.set_title('Raw Objectives (different scales)')

    ax2b = ax2.twinx()
    q_raw = [next((r['avg_travel_time'] for r in q_results if str(r['positions']) == l and r['avg_travel_time'] != float('inf')), 0) for l in all_labels]
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
        f"| Queue THRESH | {config.get_queue_param('THRESH')} |",
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
        f"| `run_config.json` | Configuration used for this run |",
        f"| `all_optimization_results.pkl` | CG equilibrium: link flows, route reconstruction, per-config results |",
        f"| `heatmap_summary.txt` | CG summary: best config, Braess paradoxes, ranking |",
        f"| `config_*/flow_heatmap.png` | Per-config CG flow heatmaps |",
        f"| `config_*/reconstruction/` | Per-config route reconstruction analysis |",
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
        f"| `objective_comparison.png` | CG vs Queue normalized objective bar chart |",
        f"| `report.md` | This report |",
        "",
    ])

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    print(f"Report saved to {report_path}")
    return report_path


def run_pipeline(config_path: str) -> str:
    """Run the complete EV charger optimization pipeline end-to-end.

    Returns: path to experiment directory.
    """
    t_start = time.time()
    config = Config.from_json(config_path)

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_dir = os.path.join("results", f"{timestamp}_n={len(config.possible_charger_positions)}_chargers={config.num_chargers}")
    os.makedirs(experiment_dir, exist_ok=True)
    config.to_json(os.path.join(experiment_dir, "run_config.json"))

    timing = {}
    cg_results = None
    queue_results = None

    # Step 1: BPR fitting
    bpr_config = config.pipeline.get("bpr_generation", {})
    rf = config.road_filter
    # Network-specific cache paths — topology changes (=prune, =merge) get their own cache
    coord_hash = int(hashlib.md5(str(config.coordinates).encode()).hexdigest()[:8], 16) % (10 ** 8)
    topo_tag = ""
    if rf.get("prune_dead_ends", True):
        topo_tag += "_pruned"
    if rf.get("merge_chains", True):
        topo_tag += "_merged"
    if rf.get("contract_edges", True):
        topo_tag += f"_ctr{rf.get('contract_threshold', 100)}"
    bpr_data_path = f"data/traffic_data_{coord_hash}{topo_tag}.csv"
    bpr_cache_path = f"data/cached_results_{coord_hash}{topo_tag}.pkl"
    # Fall back to original cache ONLY when no topology changes
    if not topo_tag:
        if not os.path.exists(bpr_cache_path) and not os.path.exists(bpr_data_path):
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
        )
        timing['bpr_fitting'] = time.time() - t0
    else:
        print("\nSkipping BPR fitting (using cache).")
        pandas_df, model_fitter = load_or_fit_model(
            data_path=bpr_data_path,
            cache_path=bpr_cache_path,
            coordinates=config.coordinates,
            bpr_config=bpr_config,
            road_filter_config=config.road_filter,
        )

    # Step 2: Congestion-game equilibrium
    if not config.pipeline.get("skip_cg_optimization", False):
        print("\n" + "=" * 80)
        print("STEP 2: Congestion-Game Equilibrium")
        print("=" * 80)
        t0 = time.time()
        od_demand = config.get_od_demand_tuples()
        rf = config.road_filter
        cg_highway_types = rf.get('highway_types') if rf.get('enabled', True) else None
        cg_prune_de = rf.get('prune_dead_ends', True)
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
            highway_types=cg_highway_types,
            prune_dead_ends=cg_prune_de,
        )
        timing['cg_optimization'] = time.time() - t0
        best_grid = grids[np.argmin([g.travel_time_obj for g in grids])]
        cg_results = {
            'best_chargers': [int(x) for x in best_grid.chargers],
            'best_objective': float(best_grid.travel_time_obj),
            'num_configs': len(grids),
            'all_configs': [
                {'chargers': [int(x) for x in g.chargers], 'objective': float(g.travel_time_obj)}
                for g in grids
            ],
            'stage_counts': grids[0].net.stage_counts if hasattr(grids[0], 'net') else {},
        }
        all_opt_path = os.path.join(experiment_dir, 'all_optimization_results.pkl')
    else:
        print("\nSkipping CG optimization.")
        timing['cg_optimization'] = 0
        all_opt_path = os.path.join(experiment_dir, 'all_optimization_results.pkl')
        if not os.path.exists(all_opt_path):
            raise FileNotFoundError(f"CG optimization skipped but {all_opt_path} not found.")

    # Step 3 & 4: Queue-based simulation
    convergence_data = None
    if not config.pipeline.get("skip_queue_simulation", False) and QUEUE_SIM_AVAILABLE:
        print("\n" + "=" * 80)
        print("STEP 3: Queue-Based NE Assignments")
        print("=" * 80)
        from queue_sim.find_nash import find_nash_assignments
        t0 = time.time()
        ne_pkl_path, convergence_data = find_nash_assignments(
            config, experiment_dir, all_opt_path
        )
        timing['queue_ne'] = time.time() - t0

        if convergence_data:
            _save_convergence_csv(convergence_data, os.path.join(experiment_dir, 'queue', 'ne_convergence.csv'))
            print(f"Convergence data saved to {experiment_dir}/queue/ne_convergence.csv")

        print("\n" + "=" * 80)
        print("STEP 4: Queue-Based Greedy vs Exhaustive Comparison")
        print("=" * 80)
        from queue_sim.comparison import run_comparison
        t0 = time.time()
        queue_results = run_comparison(
            config, experiment_dir, all_opt_path, ne_pkl_path
        )
        timing['queue_comparison'] = time.time() - t0
    elif not QUEUE_SIM_AVAILABLE:
        print(f"\nSkipping queue simulation (requires macOS: {_QUEUE_SIM_ERROR})")
        timing['queue_ne'] = 0
        timing['queue_comparison'] = 0
    else:
        print("\nSkipping queue simulation (config).")
        timing['queue_ne'] = 0
        timing['queue_comparison'] = 0

    # Step 5: Report + machine-readable summary
    print("\n" + "=" * 80)
    print("STEP 5: Report Generation")
    print("=" * 80)
    timing['total'] = time.time() - t_start
    # Read network stage counts from BPR meta or CG grid
    network_stages = None
    bpr_meta_path = bpr_data_path.replace('.csv', '_meta.json')
    if os.path.exists(bpr_meta_path):
        with open(bpr_meta_path, 'r') as f:
            meta = json.load(f)
            network_stages = meta.get('stage_counts')
    if not network_stages and cg_results and cg_results.get('stage_counts'):
        network_stages = cg_results['stage_counts']
    generate_report(experiment_dir, config, timing, cg_results, queue_results, convergence_data, network_stages)

    if cg_results and queue_results:
        _plot_objective_comparison(
            cg_results, queue_results,
            os.path.join(experiment_dir, 'objective_comparison.png')
        )

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
    }
    # Include network stage counts if available from BPR meta
    bpr_meta_path = bpr_data_path.replace('.csv', '_meta.json')
    if os.path.exists(bpr_meta_path):
        with open(bpr_meta_path, 'r') as f:
            meta = json.load(f)
            summary['network_stages'] = meta.get('stage_counts', {})
    summary_path = os.path.join(experiment_dir, 'experiment_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Experiment summary saved to {summary_path}")

    print(f"\n{'=' * 80}")
    print(f"Pipeline complete. Experiment directory: {experiment_dir}")
    print(f"Total time: {timing['total']:.1f}s")
    print(f"{'=' * 80}")

    return experiment_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run unified EV charger optimization pipeline')
    parser.add_argument('--config', type=str, default='config.json', help='Path to config JSON')
    args = parser.parse_args()
    run_pipeline(args.config)
