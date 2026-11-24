#!/usr/bin/env python3
"""Regenerate the route-recovery figures (Fig. 7 and Fig. 8) for the demand-360
run from its saved equilibrium: no CG re-solve, no BPR re-run.

* Fig. 7 (``7_flow_reconstructions.png``): progressive flow maps for
  k = 2, 4, 8, 16, 32 (original + five panels, a 3x2 grid).
* Fig. 8 (``8_flow_coverage_analysis.png``): the single-plot dual-axis format
  (blue coverage, red RMSE) used in the original manuscript.

Usage:
    python scripts/plot_k_routes_analysis.py \
        [--run-dir results/figs_demand360/figs_k_recovery_d360-5784af279095] \
        [--chargers 44,117] [--images-dir "docs/.../images"]
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import tempfile
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

K_POOL = [2, 4, 8, 16, 32, 64]   # route-pool limit stays at 64 for consistency
K_PANELS = [2, 4, 8, 16, 32]     # plotted levels (original + 5 panels = 3x2 grid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",
                        default="results/figs_demand360/figs_k_recovery_d360-5784af279095")
    parser.add_argument("--chargers", default="44,117")
    parser.add_argument("--images-dir",
                        default="docs/Optimum_EV_Placement__Journal_ (3)/images")
    args = parser.parse_args()

    from src.utils import (
        analyze_route_reconstruction, _plot_flow_on_axis, save_publication_figure)

    chargers = tuple(int(value) for value in args.chargers.split(","))
    checkpoint = os.path.join(
        args.run_dir, "cg_checkpoints",
        f"placement_{'-'.join(str(value) for value in chargers)}.pkl")
    with open(checkpoint, "rb") as handle:
        grid, _elapsed = pickle.load(handle)
    with open(os.path.join(args.run_dir, "all_optimization_results.pkl"), "rb") as handle:
        data = pickle.load(handle)
    link_flows = data["configurations"][chargers]["link_flows"]

    with tempfile.TemporaryDirectory(prefix="k-routes-") as tmp:
        reconstruction = analyze_route_reconstruction(
            network=grid, link_flows_dict=link_flows,
            k_values=list(K_POOL), save_dir=tmp)
    k_metrics = reconstruction["k_metrics"]

    def metrics_for(k):
        return k_metrics.get(k, k_metrics.get(str(k)))

    for k in sorted(int(key) for key in k_metrics):
        metrics = metrics_for(k)
        print(f"k={k:3d} coverage {metrics['coverage']:.1f}% "
              f"rmse {metrics['rmse']:.3f} mae {metrics['mae']:.3f} "
              f"corr {metrics['correlation']:.4f}")

    # --- Fig. 7: progressive flow maps (original + k = 2, 4, 8, 16, 32) ---
    original_flows = np.zeros(len(link_flows))
    for link_id, link_data in link_flows.items():
        original_flows[link_id] = link_data["total_flow"]

    n_plots = len(K_PANELS) + 1
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    plt.ioff()
    fig = plt.figure(figsize=(4.2 * n_cols, 3.8 * n_rows))
    ax = plt.subplot(n_rows, n_cols, 1)
    grid.cvxpy_link_flows = original_flows
    _plot_flow_on_axis(grid, ax, "Original Flows", original_flows)
    for idx, k in enumerate(K_PANELS, start=2):
        metrics = metrics_for(k)
        ax = plt.subplot(n_rows, n_cols, idx)
        flows = np.zeros(len(link_flows))
        for route in metrics["routes"]:
            path = route["links"]
            flow = route["flow"]
            for i in range(len(path) - 1):
                start_node, end_node = path[i], path[i + 1]
                for link_id, link_data in link_flows.items():
                    if (link_data["start_node_id"] == start_node
                            and link_data["end_node_id"] == end_node):
                        flows[link_id] += flow
                        break
        grid.cvxpy_link_flows = flows
        _plot_flow_on_axis(
            grid, ax,
            f"k={k}\nCoverage={metrics['coverage']:.1f}%\nMAE={metrics['mae']:.3f}",
            flows)
    plt.tight_layout()
    figure7 = os.path.join(args.images_dir, "7_flow_reconstructions.png")
    save_publication_figure(fig, figure7)
    plt.close(fig)
    print(f"wrote {figure7}")

    # --- Fig. 8: dual-axis coverage/RMSE plot (original manuscript format) ---
    ks = list(K_PANELS)
    coverage = [metrics_for(k)["coverage"] for k in ks]
    rmse = [metrics_for(k)["rmse"] for k in ks]

    blue, red = "#1f77b4", "#d62728"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, coverage, color=blue, marker="o", markersize=8,
            linewidth=2.5, label="Flow Coverage")
    ax.set_xlabel("Number of Routes (k)", fontsize=14)
    ax.set_ylabel("Flow Coverage (%)", color=blue, fontsize=14)
    ax.tick_params(axis="y", labelcolor=blue, labelsize=12)
    ax.tick_params(axis="x", labelsize=12)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(ks, rmse, color=red, marker="s", markersize=8,
             linewidth=2.5, label="RMSE")
    ax2.set_ylabel("RMSE", color=red, fontsize=14)
    ax2.tick_params(axis="y", labelcolor=red, labelsize=12)

    ax.set_title("Flow Coverage and RMSE vs. Number of Routes", fontsize=16)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [line.get_label() for line in lines],
              loc="center right", fontsize=13, framealpha=0.9)
    fig.tight_layout()
    figure8 = os.path.join(args.images_dir, "8_flow_coverage_analysis.png")
    fig.savefig(figure8, dpi=200)
    plt.close(fig)
    print(f"wrote {figure8}")


if __name__ == "__main__":
    main()
