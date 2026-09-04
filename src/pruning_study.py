"""Run and report a network-only pruning sweep."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
import pandas as pd
import osmnx as ox

from src.config import NetworkConfig
from src.graph_cache import get_graph
from src.network_pruning import (
    PROFILE_ORDER,
    ROAD_PROFILES,
    build_profile,
    consolidate_intersections,
    filter_highways,
    graph_metrics,
    largest_component,
    path_fidelity,
    prepare_source_graph,
    project_graph,
    crs_is_projected,
    timed_consolidation,
)


def _bbox_area_km2(coordinates):
    north, south, east, west = map(float, coordinates)
    mid_lat = math.radians((north + south) / 2.0)
    height = 111.32 * (north - south)
    width = 111.32 * math.cos(mid_lat) * (east - west)
    return abs(height * width)


def _edge_segments(graph):
    segments = []
    for u, v, _, data in graph.edges(keys=True, data=True):
        geometry = data.get("geometry")
        if geometry is not None and hasattr(geometry, "coords"):
            coords = np.asarray(geometry.coords)
            if len(coords) >= 2:
                segments.extend(np.stack([coords[:-1], coords[1:]], axis=1))
        elif u in graph and v in graph:
            segments.append(np.asarray([
                [graph.nodes[u]["x"], graph.nodes[u]["y"]],
                [graph.nodes[v]["x"], graph.nodes[v]["y"]],
            ]))
    return segments


def _draw_graph(ax, graph, title, node_size=1.5, removed_nodes=None):
    segments = _edge_segments(graph)
    if segments:
        ax.add_collection(LineCollection(segments, colors="#6f7782", linewidths=0.35, alpha=0.65))
    if graph.nodes:
        points = np.asarray([[data["x"], data["y"]] for _, data in graph.nodes(data=True)])
        ax.scatter(points[:, 0], points[:, 1], s=node_size, c="#16884a", alpha=0.8, zorder=2)
    if removed_nodes:
        points = np.asarray([
            [removed_nodes[node]["x"], removed_nodes[node]["y"]]
            for node in removed_nodes
        ])
        if len(points):
            ax.scatter(points[:, 0], points[:, 1], s=max(node_size, 2), c="#d95f02",
                       alpha=0.6, zorder=3, label="removed")
            ax.legend(loc="lower right", fontsize=7)
    ax.autoscale()
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"{title}\nN={len(graph):,}, E={graph.number_of_edges():,}", fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _save_figure(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_profile_comparison(profile_data, output_path):
    fig, axes = plt.subplots(1, len(profile_data), figsize=(6 * len(profile_data), 6))
    axes = np.atleast_1d(axes)
    for ax, profile in zip(axes, profile_data):
        _draw_graph(ax, profile_data[profile]["projected"], profile.replace("_", " "))
    fig.suptitle("Road hierarchy after topology-only simplification", fontsize=14)
    _save_figure(fig, output_path)


def _plot_radius_sweep(reference, radii, output_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, radius in zip(axes.flat, radii):
        graph, _, _ = consolidate_intersections(reference, radius)
        _draw_graph(ax, graph, f"intersection radius={radius:g} m")
    fig.suptitle("Intersection-consolidation sweep (buffer radius; pair distance can be ≈2×)", fontsize=14)
    _save_figure(fig, output_path)


def _plot_close_diagnostics(reference, summary, profile, recovered, radii, output_path):
    points = np.asarray([[data["x"], data["y"]] for _, data in reference.nodes(data=True)])
    from scipy.spatial import cKDTree
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(2, len(points)))
    nearest = np.zeros(len(points)) if len(points) == 1 else distances[:, 1]
    subset = summary[
        (summary["profile"] == profile)
        & (summary["connector_recovery"] == recovered)
        & (summary["radius_m"].isin(radii))
    ].sort_values("radius_m")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(np.clip(nearest, 0, 200), bins=40, color="#3288bd", alpha=0.85)
    axes[0].axvline(10, color="black", linestyle="--", linewidth=1)
    axes[0].axvline(20, color="black", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Nearest-node distance (m; clipped at 200)")
    axes[0].set_ylabel("Nodes")
    axes[0].set_title("Topology-only nearest neighbors")

    axes[1].plot(subset["radius_m"], subset["nodes"], marker="o", label="nodes")
    axes[1].plot(subset["radius_m"], subset["largest_scc_nodes"], marker="s", label="largest SCC")
    axes[1].set_xlabel("Consolidation radius (m)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Coarsening and directed retention")
    axes[1].legend()

    axes[2].plot(subset["radius_m"], subset["unconnected_close_pairs_le_20m"],
                 marker="o", label="unconnected pairs ≤20m")
    axes[2].plot(subset["radius_m"], subset["largest_cluster_size"],
                 marker="s", label="largest cluster size")
    axes[2].set_xlabel("Consolidation radius (m)")
    axes[2].set_title("Residual proximity and cluster size")
    axes[2].legend()
    fig.suptitle(f"Close-node diagnostics: {profile.replace('_', ' ')}", fontsize=14)
    _save_figure(fig, output_path)


def _plot_component_comparison(graph, output_path):
    wcc = largest_component(graph, strong=False)
    scc = largest_component(graph, strong=True)
    removed = {node: graph.nodes[node] for node in graph if node not in scc}
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _draw_graph(axes[0], graph, "all retained-profile components")
    _draw_graph(axes[1], wcc, "largest weak component")
    _draw_graph(axes[2], scc, "largest strong component", removed_nodes=removed)
    fig.suptitle("Connectivity policy comparison", fontsize=14)
    _save_figure(fig, output_path)


def _plot_steps(source, filtered, topology, consolidated, final_scc, output_path,
                profile, radius, recovered):
    graphs = [source, filtered, topology, consolidated, final_scc]
    labels = [
        "downloaded tertiary+ source",
        f"{profile.replace('_', ' ')} filtered",
        "topology simplified",
        f"intersections radius={radius:g} m",
        "largest SCC",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, graph, label in zip(axes.flat, graphs, labels):
        _draw_graph(ax, graph, label)
    axes.flat[-1].axis("off")
    suffix = " with connector recovery" if recovered else ""
    fig.suptitle(f"Selected diagnostic pipeline{suffix}", fontsize=14)
    _save_figure(fig, output_path)


def _rank_summary(summary):
    ranked = summary.copy()
    ranked["violation_count"] = (
        (ranked["removable_pass_through_nodes"] > 0).astype(int)
        + (ranked["invalid_edges"] > 0).astype(int)
        + (ranked["missing_provenance_edges"] > 0).astype(int)
        + (ranked["reachability_preserved_fraction"] < 1.0).astype(int)
        + (ranked["introduced_reachable_pairs"] > 0).astype(int)
        + (ranked["path_distortion_median"] > 0.02).astype(int)
        + (ranked["path_distortion_p95"] > 0.05).astype(int)
        + (ranked["scc_wcc_ratio"] < 0.90).astype(int)
        + (~ranked["topology_fidelity_valid"].astype(bool)).astype(int)
    )
    ranked = ranked.sort_values([
        "eligible",
        "violation_count",
        "nodes",
        "unconnected_close_pairs_le_20m",
        "scc_wcc_ratio",
        "path_distortion_p95",
        "total_road_km",
    ], ascending=[False, True, True, True, False, True, False], kind="mergesort")
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    return ranked


def _write_report(config, summary, leading, output_path, total_elapsed,
                  selection_mode="automated"):
    eligible = summary[summary["eligible"]]
    lines = [
        f"# Pruning study: {config.name}",
        "",
        "This report contains network construction and pruning only. No BPR,",
        "congestion-game, queue, or charger-placement experiment was run.",
        "",
        "## Input",
        "",
        f"- Coordinates `[north, south, east, west]`: `{config.coordinates}`",
        f"- Approximate bounding-box area: `{_bbox_area_km2(config.coordinates):.1f} km²`",
        f"- Total sweep wall time: `{total_elapsed:.1f} s`",
        f"- Profiles: `{config.road_filter['sweep_profiles']}`",
        f"- Intersection radii: `{config.road_filter['sweep_radii_m']} m`",
        "",
        "Consolidation uses node-buffer radius semantics. Two directly connected",
        "nodes can therefore merge at a separation up to twice the configured",
        "radius, while complete linkage prevents transitive over-expansion.",
        "",
        "## Result",
        "",
    ]
    if eligible.empty:
        lines.extend([
            "No variant passed every prespecified correctness gate. The leading",
            "diagnostic variant below is **not** a recommended final network.",
        ])
    else:
        lines.append(f"{len(eligible)} variants passed all correctness gates.")
    lines.extend([
        "",
        "## Selected diagnostic variant",
        "",
        f"- Selection mode: `{selection_mode}`",
        f"- Profile: `{leading['profile']}`",
        f"- Connector recovery: `{bool(leading['connector_recovery'])}`",
        f"- Intersection radius: `{leading['radius_m']:g} m`",
        f"- Nodes/edges: `{int(leading['nodes']):,} / {int(leading['edges']):,}`",
        f"- Largest SCC nodes: `{int(leading['largest_scc_nodes']):,}`",
        f"- SCC/WCC ratio: `{leading['scc_wcc_ratio']:.3f}`",
        f"- Median/p95 path distortion: `{leading['path_distortion_median']:.3%} / {leading['path_distortion_p95']:.3%}`",
        f"- Unconnected close pairs ≤20 m: `{int(leading['unconnected_close_pairs_le_20m'])}`",
        f"- Eligibility: `{bool(leading['eligible'])}`",
        "",
        (
            "**Warning:** this explicitly selected diagnostic variant fails at least one "
            "prespecified correctness gate and is not a validated default."
            if not bool(leading["eligible"])
            else "This selected variant passes every prespecified correctness gate."
        ),
        "",
        "## Top variants",
        "",
        "| Rank | Profile | Recovery | Radius | N | E | SCC/WCC | p95 distortion | Close ≤20m | Eligible |",
        "|---:|---|:---:|---:|---:|---:|---:|---:|---:|:---:|",
    ])
    for row in summary.head(12).itertuples():
        lines.append(
            f"| {int(row.rank)} | {row.profile} | {bool(row.connector_recovery)} | "
            f"{row.radius_m:g} | {int(row.nodes)} | {int(row.edges)} | "
            f"{row.scc_wcc_ratio:.3f} | {row.path_distortion_p95:.2%} | "
            f"{int(row.unconnected_close_pairs_le_20m)} | {bool(row.eligible)} |"
        )
    lines.extend([
        "",
        "The ranking first enforces all correctness gates, then prefers fewer",
        "nodes, fewer unexplained close pairs, greater SCC retention, lower path",
        "distortion, and greater retained road coverage. Final defaults and node",
        "targets require human review of the accompanying maps.",
    ])
    output_path.write_text("\n".join(lines) + "\n")


def _unproject(graph):
    crs = graph.graph.get("crs")
    if crs is not None and crs_is_projected(crs):
        return project_graph(graph, to_crs="EPSG:4326")
    return copy.deepcopy(graph)


def run_pruning_sweep(config_path: str) -> str:
    config = NetworkConfig.from_json(config_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = Path(config.output_dir or "results/pruning_studies") / f"{timestamp}_{config.name}"
    root.mkdir(parents=True, exist_ok=False)
    (root / "plots").mkdir()
    (root / "resolved_network_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    started = time.perf_counter()
    profiles = [profile for profile in PROFILE_ORDER if profile in config.road_filter["sweep_profiles"]]
    radii = sorted(float(value) for value in config.road_filter["sweep_radii_m"])
    union_types = sorted({road for profile in profiles for road in ROAD_PROFILES[profile]})
    north, south, east, west = config.coordinates
    source = get_graph(
        (west, south, east, north),
        highway_types=union_types,
        simplify=False,
        retain_all=True,
    )
    source = prepare_source_graph(source)
    area_km2 = _bbox_area_km2(config.coordinates)

    profile_data = {}
    rows = []
    for index, profile in enumerate(profiles):
        broader = None
        if index + 1 < len(profiles):
            broader = filter_highways(source, ROAD_PROFILES[profiles[index + 1]])
        built = build_profile(
            source,
            profile,
            connector_source=broader,
            connector_threshold=float(config.road_filter["connector_threshold"]),
        )
        profile_data[profile] = {
            "projected": built["projected"],
            "recovered_projected": built["recovered_projected"],
            "connector_metadata": built["connector_metadata"],
        }
        choices = [(False, built["projected"])]
        if built["recovered_projected"] is not None:
            choices.append((True, built["recovered_projected"]))
        for recovered, reference in choices:
            variant_filtered = (
                built["recovered_filtered"] if recovered else built["filtered"]
            )
            variant_topology = (
                built["recovered_topology"] if recovered else built["topology"]
            )
            topology_fidelity = path_fidelity(
                variant_filtered,
                variant_topology,
                {node: node for node in variant_topology.nodes},
                seed=int(config.road_filter["diagnostic_seed"]),
            )
            for radius in radii:
                graph, _, _, fidelity, elapsed = timed_consolidation(
                    reference, radius, seed=int(config.road_filter["diagnostic_seed"])
                )
                metrics = graph_metrics(
                    graph,
                    area_km2=area_km2,
                    profile=profile,
                    radius_m=radius,
                    connector_recovery=recovered,
                    fidelity=fidelity,
                    elapsed_seconds=elapsed,
                )
                connector_meta = built["connector_metadata"] if recovered else {}
                metrics.update({
                    "source_nodes": len(source),
                    "source_edges": source.number_of_edges(),
                    "filtered_nodes": len(variant_filtered),
                    "filtered_edges": variant_filtered.number_of_edges(),
                    "topology_nodes": len(variant_topology),
                    "topology_edges": variant_topology.number_of_edges(),
                    "consolidated_nodes": len(graph),
                    "consolidated_edges": graph.number_of_edges(),
                })
                metrics.update({
                    f"topology_{key}": value
                    for key, value in topology_fidelity.items()
                })
                topology_valid = bool(
                    topology_fidelity["reachability_preserved_fraction"] >= 1.0
                    and topology_fidelity["introduced_reachable_pairs"] == 0
                    and topology_fidelity["path_distortion_p95"] <= 1e-9
                )
                metrics["topology_fidelity_valid"] = topology_valid
                metrics["eligible"] = bool(metrics["eligible"] and topology_valid)
                metrics["connector_paths_added"] = int(connector_meta.get("paths_added", 0))
                metrics["connector_edges_added"] = int(connector_meta.get("edges_added", 0))
                rows.append(metrics)

    summary = _rank_summary(pd.DataFrame(rows))
    summary.to_csv(root / "pruning_summary.csv", index=False)
    automated_leading = summary.iloc[0]
    leading = automated_leading
    selection_mode = "automated correctness-gated ranking"
    requested_profile = config.road_filter.get("diagnostic_profile")
    requested_radius = config.road_filter.get("diagnostic_radius_m")
    requested_recovery = bool(
        config.road_filter.get("diagnostic_connector_recovery", False)
    )
    if requested_profile is not None or requested_radius is not None:
        requested_profile = requested_profile or str(automated_leading["profile"])
        requested_radius = float(
            automated_leading["radius_m"] if requested_radius is None else requested_radius
        )
        matches = summary[
            (summary["profile"] == requested_profile)
            & np.isclose(summary["radius_m"], requested_radius)
            & (summary["connector_recovery"].astype(bool) == requested_recovery)
        ]
        if matches.empty:
            raise ValueError(
                "requested diagnostic profile/radius/recovery was not produced by the sweep"
            )
        leading = matches.iloc[0]
        selection_mode = "explicit configuration override (may be ineligible)"
    profile = str(leading["profile"])
    recovered = bool(leading["connector_recovery"])
    radius = float(leading["radius_m"])
    selected_index = profiles.index(profile)
    selected_broader = None
    if selected_index + 1 < len(profiles):
        selected_broader = filter_highways(source, ROAD_PROFILES[profiles[selected_index + 1]])
    built = build_profile(
        source,
        profile,
        connector_source=selected_broader,
        connector_threshold=float(config.road_filter["connector_threshold"]),
    )
    reference = built["recovered_projected"] if recovered else built["projected"]
    filtered = built["recovered_filtered"] if recovered else built["filtered"]
    topology = built["recovered_topology"] if recovered else built["topology"]
    selected_graph, _, _ = consolidate_intersections(reference, radius)
    final_scc = largest_component(selected_graph, strong=True)

    _plot_profile_comparison(profile_data, root / "plots" / "road_profile_comparison.png")
    _plot_radius_sweep(reference, radii, root / "plots" / "consolidation_sweep.png")
    _plot_close_diagnostics(
        reference, summary, profile, recovered, radii,
        root / "plots" / "close_node_diagnostics.png",
    )
    _plot_component_comparison(selected_graph, root / "plots" / "component_comparison.png")
    _plot_steps(
        source,
        _unproject(filtered) if filtered is not None else source,
        _unproject(topology) if topology is not None else source,
        selected_graph,
        final_scc,
        root / "plots" / "pruning_steps.png",
        profile,
        radius,
        recovered,
    )

    from src.road_network import RoadNet
    selected_unprojected = _unproject(final_scc)
    road_net = RoadNet(config.name)
    road_net.graph = selected_unprojected
    road_net.stage_counts = {
        "source": {"nodes": len(source), "edges": source.number_of_edges()},
        "filtered": {"nodes": len(filtered), "edges": filtered.number_of_edges()},
        "topology": {"nodes": len(topology), "edges": topology.number_of_edges()},
        "consolidated": {"nodes": len(selected_graph), "edges": selected_graph.number_of_edges()},
        "final_scc": {"nodes": len(final_scc), "edges": final_scc.number_of_edges()},
    }
    road_net.rearrange_data()
    road_net.export_artifact(
        root / "selected_network",
        source={
            "coordinates": config.coordinates,
            "profile": profile,
            "intersection_radius_m": radius,
            "connector_recovery": recovered,
            "selection_is_eligible": bool(leading["eligible"]),
        },
    )
    total_elapsed = time.perf_counter() - started
    _write_report(
        config, summary, leading, root / "pruning_report.md", total_elapsed,
        selection_mode=selection_mode,
    )
    (root / "run_manifest.json").write_text(json.dumps({
        "study": "network_pruning_only",
        "city": config.name,
        "total_elapsed_seconds": total_elapsed,
        "leading_variant": {
            "profile": profile,
            "intersection_radius_m": radius,
            "connector_recovery": recovered,
            "eligible": bool(leading["eligible"]),
        },
        "selection_mode": selection_mode,
        "automated_leading_variant": {
            "profile": str(automated_leading["profile"]),
            "intersection_radius_m": float(automated_leading["radius_m"]),
            "connector_recovery": bool(automated_leading["connector_recovery"]),
            "eligible": bool(automated_leading["eligible"]),
        },
        "outputs": {
            "summary": "pruning_summary.csv",
            "report": "pruning_report.md",
            "plots": sorted(path.name for path in (root / "plots").iterdir()),
            "selected_network": "selected_network/network_manifest.json",
        },
    }, indent=2, sort_keys=True) + "\n")
    print(f"Pruning study complete: {root}")
    return str(root)
