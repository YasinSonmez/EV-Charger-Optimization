"""Deterministic experiment scenarios derived from a canonical road graph."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np

from src.network_coarsening import select_charger_candidate_layer
from src.network_pruning import project_graph


@dataclass(frozen=True)
class GeneratedScenario:
    candidate_node_ids: list[int]
    od_demand: dict[str, list[int]]
    metadata: dict[str, Any]


def _canonical_graph(road_net) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(crs="EPSG:4326")
    for row in road_net.nodes.itertuples():
        graph.add_node(
            int(row.node_id), x=float(row.lon), y=float(row.lat),
            lon=float(row.lon), lat=float(row.lat),
            node_osmid=getattr(row, "node_osmid", None),
        )
    for row in road_net.edges.itertuples():
        graph.add_edge(
            int(row.start_node_id), int(row.end_node_id),
            key=int(row.link_id), link_id=int(row.link_id),
            length=float(row.length),
            travel_time=float(row.travel_time),
            highway=str(row.type).split("|"),
        )
    return graph


def _boundary_nodes(graph: nx.MultiDiGraph, excluded: set[int], limit: int) -> list[int]:
    nodes = [node for node in graph if node not in excluded]
    if len(nodes) < 2:
        raise ValueError("at least two non-candidate nodes are required for OD generation")
    points = np.asarray([[graph.nodes[node]["x"], graph.nodes[node]["y"]] for node in nodes])
    center = np.mean(points, axis=0)
    radii = np.linalg.norm(points - center, axis=1)
    order = sorted(
        range(len(nodes)),
        key=lambda index: (-float(radii[index]), str(nodes[index])),
    )
    return [nodes[index] for index in order[: min(int(limit), len(nodes))]]


def _pair_score(graph: nx.MultiDiGraph, source: int, target: int) -> tuple[float, float]:
    a = graph.nodes[source]
    b = graph.nodes[target]
    separation = math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))
    travel_time = float(nx.shortest_path_length(graph, source, target, weight="travel_time"))
    return separation, travel_time


def _select_od_pairs(
    graph: nx.MultiDiGraph,
    candidates: list[int],
    *,
    count: int,
    boundary_pool_size: int,
) -> list[tuple[int, int]]:
    boundary = _boundary_nodes(graph, set(candidates), boundary_pool_size)
    ranked = []
    for source in boundary:
        for target in boundary:
            if source == target:
                continue
            try:
                score = _pair_score(graph, source, target)
                # The final SCC should make these checks cheap and true.  Keep
                # them explicit so scenario generation fails on a bad graph.
                if any(
                    not nx.has_path(graph, source, candidate)
                    or not nx.has_path(graph, candidate, target)
                    for candidate in candidates
                ):
                    continue
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            ranked.append((score, str(source), str(target), source, target))
    ranked.sort(key=lambda item: (-item[0][0], -item[0][1], item[1], item[2]))
    selected = []
    used = set()
    for _score, _source_key, _target_key, source, target in ranked:
        if source in used or target in used:
            continue
        selected.append((source, target))
        used.update((source, target))
        if len(selected) == int(count):
            return selected
    raise ValueError(
        f"could not select {count} disjoint feasible OD pair(s) from "
        f"{len(boundary)} boundary nodes"
    )


def generate_scenario(road_net, settings: dict) -> GeneratedScenario:
    """Select candidates and OD demand without requiring canonical node IDs."""
    graph = _canonical_graph(road_net)
    if not nx.is_strongly_connected(graph):
        raise ValueError("scenario generation requires a strongly connected graph")
    projected = project_graph(graph)
    candidates, candidate_metadata, diagnostics = select_charger_candidate_layer(
        projected,
        max_candidates=int(settings["candidate_count"]),
        interchange_merge_diameter_m=float(
            settings.get("interchange_merge_diameter_m", 250.0)
        ),
    )
    candidates = [int(node) for node in candidates]
    od_pairs = _select_od_pairs(
        projected,
        candidates,
        count=int(settings.get("od_pair_count", 1)),
        boundary_pool_size=int(settings.get("boundary_pool_size", 64)),
    )
    demand = settings.get("demand", {})
    od_demand = {
        f"{int(origin)},{int(destination)}": [
            int(demand.get("F1", 0)), int(demand.get("F2", 0))
        ]
        for origin, destination in od_pairs
    }
    metadata = {
        "strategy": {
            "candidates": "interchanges_then_farthest_point",
            "od": "boundary_max_separation",
        },
        "seed": int(settings.get("seed", 42)),
        "candidate_diagnostics": diagnostics,
        "candidates": [
            {
                "node_id": node,
                "lat": float(graph.nodes[node]["lat"]),
                "lon": float(graph.nodes[node]["lon"]),
                "kind": candidate_metadata[node]["kind"],
            }
            for node in candidates
        ],
        "od_pairs": [
            {
                "origin": int(origin),
                "destination": int(destination),
                "origin_lat": float(graph.nodes[origin]["lat"]),
                "origin_lon": float(graph.nodes[origin]["lon"]),
                "destination_lat": float(graph.nodes[destination]["lat"]),
                "destination_lon": float(graph.nodes[destination]["lon"]),
                "free_flow_seconds": float(nx.shortest_path_length(
                    graph, origin, destination, weight="travel_time"
                )),
            }
            for origin, destination in od_pairs
        ],
    }
    return GeneratedScenario(candidates, od_demand, metadata)


def plot_scenario(road_net, scenario: GeneratedScenario, output_path: str) -> None:
    graph = _canonical_graph(road_net)
    segments = []
    for source, target in graph.edges():
        a, b = graph.nodes[source], graph.nodes[target]
        segments.append([(a["lon"], a["lat"]), (b["lon"], b["lat"])])
    fig, ax = plt.subplots(figsize=(9, 9))
    if segments:
        ax.add_collection(LineCollection(segments, colors="#7d8b99", linewidths=0.55, alpha=0.65))
    candidates = scenario.candidate_node_ids
    ax.scatter(
        [graph.nodes[node]["lon"] for node in candidates],
        [graph.nodes[node]["lat"] for node in candidates],
        marker="*", s=110, color="#159447", edgecolors="black", label="charger candidates", zorder=4,
    )
    for record in scenario.metadata["od_pairs"]:
        origin, destination = record["origin"], record["destination"]
        route = nx.shortest_path(graph, origin, destination, weight="travel_time")
        route_segments = [
            [(graph.nodes[a]["lon"], graph.nodes[a]["lat"]),
             (graph.nodes[b]["lon"], graph.nodes[b]["lat"])]
            for a, b in zip(route[:-1], route[1:])
        ]
        ax.add_collection(LineCollection(route_segments, colors="#d62728", linewidths=2.0, alpha=0.85))
        ax.scatter([graph.nodes[origin]["lon"]], [graph.nodes[origin]["lat"]], marker="o", s=75,
                   color="#1f77b4", edgecolors="black", label="OD origin", zorder=5)
        ax.scatter([graph.nodes[destination]["lon"]], [graph.nodes[destination]["lat"]], marker="s", s=75,
                   color="#ff7f0e", edgecolors="black", label="OD destination", zorder=5)
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(
        f"Generated scenario: N={graph.number_of_nodes():,}, E={graph.number_of_edges():,}, "
        f"candidates={len(candidates)}"
    )
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
