#!/usr/bin/env python3
"""Plot the CG and queue-based NE link-flow heatmaps side by side with one
shared color scale, for one charger configuration.

The queue flows are rebuilt from the saved better-response route counts
(``NE_path_assignments.pkl``) mapped onto the exact route library the queue
stage used (``_prune_flow_data`` with balanced quotas).  The congestion-game
flows come from ``all_optimization_results.pkl``.  Both panels share a single
colorbar and use the same style: viridis link coloring, linewidth scaled by
sqrt(flow), faint low-flow links, node dots, colored origin/destination
markers without text annotations, and charger stars.  Also prints CG-vs-queue
link-flow Pearson/Spearman.

Usage:
    python scripts/plot_queue_ne_heatmap.py --chargers 44,159 \
        --output "docs/.../images/nash_flows_44-159.png"
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COLORS = {"light": "#b0b0b0", "blue": "#1f77b4",
          "orange": "#e8a33d", "green": "#2ca02c"}
MIN_LW, MAX_LW = 0.45, 4.0


def _draw_panel(ax, net, flows, vmax, title):
    edges = net.edges.sort_values("link_id").copy()
    flows = np.asarray(flows, dtype=float)
    if len(flows) != len(edges):
        raise ValueError(f"flow length {len(flows)} != edges {len(edges)}")
    edges["flow"] = flows
    edges["geometry"] = gpd.GeoSeries.from_wkt(edges["geometry"])
    gdf = gpd.GeoDataFrame(edges, geometry="geometry")

    norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("viridis")
    scaled = np.sqrt(np.clip(gdf["flow"].to_numpy(dtype=float) / vmax, 0, 1))
    linewidths = MIN_LW + scaled * (MAX_LW - MIN_LW)
    colors = [cmap(norm(value)) for value in gdf["flow"]]
    for geom, color, lw, flow in zip(gdf["geometry"], colors, linewidths,
                                     gdf["flow"]):
        x, y = geom.xy
        if flow < 1.0:
            faint = list(color)
            faint[3] = 0.3
            ax.plot(x, y, color=faint, linewidth=MIN_LW)
        else:
            ax.plot(x, y, color=color, linewidth=lw, solid_capstyle="round")

    node_size = max(8.0, min(20.0, 1200.0 / max(1, len(net.nodes))))
    ax.scatter(net.nodes["lon"], net.nodes["lat"], color=COLORS["light"],
               s=node_size, edgecolors="white", linewidths=0.4, zorder=3,
               label="Network nodes")
    origin, destination = (76, 120)
    ax.scatter(net.nodes.at[origin, "lon"], net.nodes.at[origin, "lat"],
               c=COLORS["blue"], s=80, edgecolors="white", linewidths=0.8,
               zorder=4, label="Origin")
    ax.scatter(net.nodes.at[destination, "lon"], net.nodes.at[destination, "lat"],
               c=COLORS["orange"], s=80, marker="s", edgecolors="white",
               linewidths=0.8, zorder=4, label="Destination")
    for index, charger_id in enumerate(sorted(net.chargers)):
        ax.scatter(net.nodes.at[charger_id, "lon"], net.nodes.at[charger_id, "lat"],
                   marker="*", s=140, color=COLORS["green"], edgecolors="white",
                   linewidths=0.8, zorder=6,
                   label="Selected charger" if index == 0 else None)
    ax.set_title(title, fontsize=11)
    ax.set_axis_off()
    return cmap, norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",
                        default="results/suite_new6_500/runs/sensitivity_base-ff8efbae0bc5")
    parser.add_argument("--chargers", default="44,159")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--output", default=None)
    parser.add_argument("--correlations-only", action="store_true")
    args = parser.parse_args()

    from src.road_network import RoadNet
    from queue_sim.find_nash import _prune_flow_data

    run = args.run_dir
    chargers = tuple(int(value) for value in args.chargers.split(","))
    ne_key = ",".join(str(value) for value in chargers)

    with open(os.path.join(run, "all_optimization_results.pkl"), "rb") as handle:
        data = pickle.load(handle)
    with open(os.path.join(run, "queue", "NE_path_assignments.pkl"), "rb") as handle:
        ne_assignments = pickle.load(handle)
    assignments = ne_assignments[ne_key]["assignments"]

    flow_data = _prune_flow_data(data, chargers, args.k, balanced=True)

    queue_flows = defaultdict(float)
    totals = {}
    for vehicle_type, group_key in (("F1", "no charging type"),
                                    ("F2", "charging type")):
        for od, group in flow_data.items():
            routes = group[group_key]
            counts = assignments[vehicle_type].get(od, [])
            if len(counts) != len(routes):
                raise ValueError(
                    f"count/library mismatch for {vehicle_type} {od}: "
                    f"{len(counts)} vs {len(routes)}")
            for count, route in zip(counts, routes):
                if count <= 0:
                    continue
                totals[vehicle_type] = totals.get(vehicle_type, 0) + int(count)
                for link_id in route.get("link_ids", []):
                    queue_flows[int(link_id)] += int(count)
    print(f"vehicle totals: {totals}")

    net = RoadNet("plot")
    net.load_artifact(os.path.join(run, "network"))
    net.chargers = chargers

    cg_link_flows = data["configurations"][chargers]["link_flows"]
    n_links = len(net.edges)
    cg_array = np.zeros(n_links, dtype=float)
    queue_array = np.zeros(n_links, dtype=float)
    for link_id, record in cg_link_flows.items():
        if int(link_id) < n_links:
            cg_array[int(link_id)] = float(record.get("total_flow", 0.0))
    for link_id, value in queue_flows.items():
        if int(link_id) < n_links:
            queue_array[int(link_id)] = float(value)

    pearson = float(np.corrcoef(cg_array, queue_array)[0, 1])
    ranks_cg = pd.Series(cg_array).rank(method="average").to_numpy()
    ranks_q = pd.Series(queue_array).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(ranks_cg, ranks_q)[0, 1])
    active = queue_array > 0
    n_active = int(active.sum())
    top_cg = set(np.argsort(-cg_array)[:n_active])
    overlap = len(top_cg & set(np.flatnonzero(active)))
    print(f"link-flow Pearson {pearson:.3f} / Spearman {spearman:.3f} over "
          f"{n_links} road links; queue-active {n_active}, top-{n_active} CG "
          f"overlap {overlap}/{n_active}")
    if args.correlations_only:
        return

    vmax = max(1e-6, float(max(cg_array.max(), queue_array.max())))
    fig = plt.figure(figsize=(16, 7.5))
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 0.045), wspace=0.02)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    colorbar_axis = fig.add_subplot(grid[0, 2])
    cmap, norm = _draw_panel(axes[0], net, cg_array, vmax,
                             "Congestion Game (Nash equilibrium)")
    _draw_panel(axes[1], net, queue_array, vmax,
                "Queuing-Based Model (approximate NE)")
    scalar = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array(np.array([vmax]))
    colorbar = fig.colorbar(scalar, cax=colorbar_axis, orientation="vertical")
    colorbar.set_label("Equilibrium link flow (vehicles)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", frameon=True,
               facecolor="white", edgecolor=COLORS["light"], ncol=2)
    fig.subplots_adjust(top=0.84, left=0.01, right=0.955, bottom=0.01)

    output = args.output or os.path.join(
        "docs/Optimum_EV_Placement__Journal_ (3)/images",
        f"nash_flows_{ne_key.replace(',', '-')}.png")
    fig.savefig(output, dpi=200)
    plt.close(fig)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
