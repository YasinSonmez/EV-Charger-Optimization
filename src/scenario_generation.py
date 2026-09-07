"""Deterministic experiment scenarios derived from a canonical road graph."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

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
        lanes = float(getattr(row, "lanes", 1.0))
        capacity = float(getattr(row, "capacity", lanes * 1000.0))
        graph.add_edge(
            int(row.start_node_id), int(row.end_node_id),
            key=int(row.link_id), link_id=int(row.link_id),
            length=float(row.length),
            travel_time=float(row.travel_time),
            capacity=capacity,
            lanes=lanes,
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


def _select_corridor_candidates(
    graph: nx.MultiDiGraph,
    od_pairs: list[tuple[int, int]],
    *,
    count: int,
    max_detour_ratio: float,
    corridor_radius_m: float,
    interchange_merge_diameter_m: float,
):
    """Select interchange-aware candidates near one or more OD corridors.

    A node is in an OD corridor when routing through it adds no more than the
    configured free-flow detour. Candidate slots are first divided among ODs,
    then any unfilled slots are selected from the union of all corridors.
    """
    reverse = graph.reverse(copy=False)
    contexts = []
    endpoints = {node for pair in od_pairs for node in pair}
    for origin, destination in od_pairs:
        baseline = float(nx.shortest_path_length(
            graph, origin, destination, weight="travel_time"
        ))
        from_origin = nx.single_source_dijkstra_path_length(
            graph, origin, weight="travel_time"
        )
        to_destination = nx.single_source_dijkstra_path_length(
            reverse, destination, weight="travel_time"
        )
        route = nx.shortest_path(
            graph, origin, destination, weight="travel_time"
        )
        route_line = LineString([
            (float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))
            for node in route
        ])
        ratios = {
            node: (float(from_origin[node]) + float(to_destination[node])) / baseline
            for node in graph
            if node not in endpoints
            and node in from_origin and node in to_destination
        }
        distances = {
            node: float(Point(
                float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])
            ).distance(route_line))
            for node in ratios
        }
        eligible = {
            node for node, ratio in ratios.items()
            if ratio <= float(max_detour_ratio) + 1e-12
            and distances[node] <= float(corridor_radius_m) + 1e-9
        }
        contexts.append({
            "od": (origin, destination), "baseline": baseline,
            "ratios": ratios, "distances": distances, "eligible": eligible,
        })

    selected = []
    candidate_metadata = {}
    base_quota, remainder = divmod(int(count), len(contexts))
    for index, context in enumerate(contexts):
        quota = base_quota + (1 if index < remainder else 0)
        quota = min(quota, len(context["eligible"]))
        if quota <= 0:
            continue
        local_graph = graph.subgraph(context["eligible"]).copy()
        local, local_metadata, _ = select_charger_candidate_layer(
            local_graph,
            max_candidates=quota,
            interchange_merge_diameter_m=interchange_merge_diameter_m,
        )
        for node in local:
            if node not in candidate_metadata:
                selected.append(node)
                candidate_metadata[node] = local_metadata[node]

    eligible_union = set().union(*(context["eligible"] for context in contexts))
    if len(selected) < int(count) and eligible_union:
        union_graph = graph.subgraph(eligible_union).copy()
        fill_count = min(len(eligible_union), int(count) + len(selected))
        fill, fill_metadata, _ = select_charger_candidate_layer(
            union_graph,
            max_candidates=fill_count,
            interchange_merge_diameter_m=interchange_merge_diameter_m,
        )
        for node in fill:
            if node in candidate_metadata:
                continue
            selected.append(node)
            candidate_metadata[node] = fill_metadata[node]
            if len(selected) == int(count):
                break

    if len(selected) < int(count):
        raise ValueError(
            f"only {len(selected)} distinct candidate nodes satisfy the "
            f"maximum detour ratio {max_detour_ratio}; requested {count}"
        )

    for node in selected:
        ratios = {
            f"{origin},{destination}": float(context["ratios"][node])
            for context in contexts
            for origin, destination in [context["od"]]
        }
        closest_od = min(ratios, key=ratios.get)
        candidate_metadata[node].update({
            "detour_ratios": ratios,
            "minimum_detour_ratio": float(ratios[closest_od]),
            "corridor_distances_m": {
                f"{context['od'][0]},{context['od'][1]}": float(
                    context["distances"][node]
                )
                for context in contexts
            },
            "minimum_corridor_distance_m": float(min(
                context["distances"][node] for context in contexts
            )),
            "closest_od": closest_od,
        })

    diagnostics = {
        "candidate_count": len(selected),
        "eligible_candidate_nodes": len(eligible_union),
        "maximum_detour_ratio": float(max_detour_ratio),
        "corridor_radius_m": float(corridor_radius_m),
        "selected_minimum_detour_ratio_max": float(max(
            candidate_metadata[node]["minimum_detour_ratio"] for node in selected
        )),
        "per_od": {
            f"{context['od'][0]},{context['od'][1]}": {
                "baseline_free_flow_seconds": float(context["baseline"]),
                "eligible_candidate_nodes": len(context["eligible"]),
                "selected_candidates_within_limit": sum(
                    context["ratios"][node] <= float(max_detour_ratio) + 1e-12
                    for node in selected
                ),
            }
            for context in contexts
        },
    }
    return selected, candidate_metadata, diagnostics


def _select_ranked_detour_candidates(
    graph: nx.MultiDiGraph,
    od_pairs: list[tuple[int, int]],
    *,
    count: int,
    min_detour_percent: float = 5.0,
    max_detour_percent: float = 20.0,
):
    """Choose evenly spaced ranks from nodes in a bounded detour interval.

    For multiple ODs, a node's score is its smallest free-flow detour over all
    OD pairs. Thus each candidate is ranked relative to the OD it can serve
    most directly, while remaining deterministic and independent of node IDs.
    """
    reverse = graph.reverse(copy=False)
    endpoints = {node for pair in od_pairs for node in pair}
    contexts = []
    for origin, destination in od_pairs:
        baseline = float(nx.shortest_path_length(
            graph, origin, destination, weight="travel_time"
        ))
        from_origin = nx.single_source_dijkstra_path_length(
            graph, origin, weight="travel_time"
        )
        to_destination = nx.single_source_dijkstra_path_length(
            reverse, destination, weight="travel_time"
        )
        route = nx.shortest_path(graph, origin, destination, weight="travel_time")
        route_line = LineString([
            (float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"]))
            for node in route
        ])
        ratios = {
            node: (float(from_origin[node]) + float(to_destination[node])) / baseline
            for node in graph
            if node not in endpoints and node in from_origin and node in to_destination
        }
        distances = {
            node: float(Point(
                float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])
            ).distance(route_line))
            for node in ratios
        }
        contexts.append({
            "od": (origin, destination), "baseline": baseline,
            "ratios": ratios, "distances": distances,
        })

    lower = 1.0 + float(min_detour_percent) / 100.0
    upper = 1.0 + float(max_detour_percent) / 100.0
    ranked = []
    for node in graph:
        if node in endpoints or any(node not in context["ratios"] for context in contexts):
            continue
        ratios = [float(context["ratios"][node]) for context in contexts]
        score = min(ratios)
        if lower - 1e-12 <= score <= upper + 1e-12:
            ranked.append((score, str(node), node))
    ranked.sort(key=lambda item: (item[0], item[1]))
    if len(ranked) < int(count):
        raise ValueError(
            f"only {len(ranked)} candidate nodes have minimum OD detour in "
            f"[{min_detour_percent}%, {max_detour_percent}%]; requested {count}"
        )

    if int(count) == 1:
        selected_ranks = [(len(ranked) - 1) // 2]
    else:
        selected_ranks = np.rint(
            np.linspace(0, len(ranked) - 1, int(count))
        ).astype(int).tolist()
    selected = [ranked[index][2] for index in selected_ranks]
    candidate_metadata = {}
    for rank, node in zip(selected_ranks, selected):
        ratios = {
            f"{context['od'][0]},{context['od'][1]}": float(context["ratios"][node])
            for context in contexts
        }
        closest_od = min(ratios, key=ratios.get)
        distances = {
            f"{context['od'][0]},{context['od'][1]}": float(context["distances"][node])
            for context in contexts
        }
        candidate_metadata[node] = {
            "kind": "ranked_detour",
            "detour_rank": int(rank),
            "detour_rank_fraction": (
                0.5 if len(ranked) == 1 else float(rank) / (len(ranked) - 1)
            ),
            "detour_ratios": ratios,
            "minimum_detour_ratio": float(ratios[closest_od]),
            "corridor_distances_m": distances,
            "minimum_corridor_distance_m": float(min(distances.values())),
            "closest_od": closest_od,
        }

    diagnostics = {
        "candidate_count": len(selected),
        "eligible_candidate_nodes": len(ranked),
        "minimum_detour_percent": float(min_detour_percent),
        "maximum_detour_percent": float(max_detour_percent),
        "selection": "ranked_linear",
        "selected_ranks": selected_ranks,
        "selected_rank_fractions": [
            candidate_metadata[node]["detour_rank_fraction"] for node in selected
        ],
        "selected_detour_percents": [
            100.0 * (candidate_metadata[node]["minimum_detour_ratio"] - 1.0)
            for node in selected
        ],
        "per_od": {
            f"{context['od'][0]},{context['od'][1]}": {
                "baseline_free_flow_seconds": float(context["baseline"]),
                "selected_candidates_within_limit": sum(
                    lower - 1e-12 <= context["ratios"][node] <= upper + 1e-12
                    for node in selected
                ),
            }
            for context in contexts
        },
    }
    return selected, candidate_metadata, diagnostics


def _path_edges(graph: nx.MultiDiGraph, path: list[int]) -> list[tuple[int, int, int]]:
    edges = []
    for source, target in zip(path[:-1], path[1:]):
        key = min(
            graph[source][target],
            key=lambda value: float(graph[source][target][value].get("travel_time", math.inf)),
        )
        edges.append((source, target, key))
    return edges


def _route_edges(graph, source, target, via=None):
    if via is None:
        return _path_edges(
            graph, nx.shortest_path(graph, source, target, weight="travel_time")
        )
    first = nx.shortest_path(graph, source, via, weight="travel_time")
    second = nx.shortest_path(graph, via, target, weight="travel_time")
    return _path_edges(graph, first) + _path_edges(graph, second)


def _has_bypass(graph, source, target, edge, via=None):
    reduced = nx.subgraph_view(
        graph, filter_edge=lambda u, v, key: (u, v, key) != edge
    )
    try:
        if via is None:
            return nx.has_path(reduced, source, target)
        return nx.has_path(reduced, source, via) and nx.has_path(reduced, via, target)
    except nx.NodeNotFound:
        return False


def saturation_diagnostics(
    graph: nx.MultiDiGraph,
    od_pairs: list[tuple[int, int]],
    candidates: list[int],
    demand: dict,
    calibration_window_hours: float,
) -> dict[str, Any]:
    """Cheap all-or-nothing offered-load check, not a simulation result."""
    scenarios = []
    f1 = int(demand.get("F1", 0))
    f2 = int(demand.get("F2", 0))
    for candidate in candidates:
        loads = defaultdict(float)
        bypassable = set()
        for source, target in od_pairs:
            f1_edges = _route_edges(graph, source, target)
            f2_edges = _route_edges(graph, source, target, via=candidate)
            for edge in f1_edges:
                loads[edge] += f1
                if _has_bypass(graph, source, target, edge):
                    bypassable.add(edge)
            for edge in f2_edges:
                loads[edge] += f2
                if _has_bypass(graph, source, target, edge, via=candidate):
                    bypassable.add(edge)
        records = []
        for edge, load in loads.items():
            capacity = float(graph.edges[edge].get("capacity", math.nan))
            window_capacity = capacity * float(calibration_window_hours)
            ratio = load / window_capacity if window_capacity > 0 else math.inf
            records.append((ratio, edge, load, window_capacity, edge in bypassable))
        usable = [record for record in records if record[-1]]
        best = max(usable, default=(0.0, None, 0.0, math.nan, False))
        scenarios.append({
            "candidate": int(candidate),
            "maximum_bypassable_vc": float(best[0]),
            "critical_link_id": (
                int(graph.edges[best[1]].get("link_id")) if best[1] else None
            ),
            "critical_link_load": float(best[2]),
            "critical_window_capacity": float(best[3]),
        })
    values = [item["maximum_bypassable_vc"] for item in scenarios]
    return {
        "method": "free_flow_all_or_nothing_bypassable_link",
        "calibration_window_hours": float(calibration_window_hours),
        "demand_per_od": {"F1": f1, "F2": f2},
        "candidate_scenarios": scenarios,
        "minimum_candidate_maximum_bypassable_vc": float(min(values, default=0.0)),
        "maximum_candidate_maximum_bypassable_vc": float(max(values, default=0.0)),
        "interpretation": "static offered-load screen; not realized equilibrium flow",
    }


def generate_scenario(
    road_net, settings: dict, calibration_window_hours: float = 0.1
) -> GeneratedScenario:
    """Select candidates and OD demand without requiring canonical node IDs."""
    graph = _canonical_graph(road_net)
    if not nx.is_strongly_connected(graph):
        raise ValueError("scenario generation requires a strongly connected graph")
    projected = project_graph(graph)
    strategy = settings.get(
        "candidate_strategy", "interchanges_then_farthest_point"
    )
    candidate_count = int(settings["candidate_count"])
    merge_diameter = float(settings.get("interchange_merge_diameter_m", 250.0))
    od_count = int(settings.get("od_pair_count", 1))
    boundary_pool_size = int(settings.get("boundary_pool_size", 64))
    if strategy == "interchanges_then_farthest_point":
        candidates, candidate_metadata, diagnostics = select_charger_candidate_layer(
            projected,
            max_candidates=candidate_count,
            interchange_merge_diameter_m=merge_diameter,
        )
        od_pairs = _select_od_pairs(
            projected, candidates, count=od_count,
            boundary_pool_size=boundary_pool_size,
        )
    elif strategy == "od_corridor_interchanges":
        od_pairs = _select_od_pairs(
            projected, [], count=od_count,
            boundary_pool_size=boundary_pool_size,
        )
        candidates, candidate_metadata, diagnostics = _select_corridor_candidates(
            projected, od_pairs, count=candidate_count,
            max_detour_ratio=float(settings.get("candidate_max_detour_ratio", 1.10)),
            corridor_radius_m=float(settings.get("candidate_corridor_radius_m", 500.0)),
            interchange_merge_diameter_m=merge_diameter,
        )
    elif strategy == "od_detour_ranked":
        od_pairs = _select_od_pairs(
            projected, [], count=od_count,
            boundary_pool_size=boundary_pool_size,
        )
        candidates, candidate_metadata, diagnostics = _select_ranked_detour_candidates(
            projected, od_pairs, count=candidate_count,
            min_detour_percent=float(settings.get("candidate_min_detour_percent", 5.0)),
            max_detour_percent=float(settings.get("candidate_max_detour_percent", 20.0)),
        )
    else:
        raise ValueError(f"unsupported candidate strategy: {strategy}")
    candidates = [int(node) for node in candidates]
    demand = settings.get("demand", {})
    od_demand = {
        f"{int(origin)},{int(destination)}": [
            int(demand.get("F1", 0)), int(demand.get("F2", 0))
        ]
        for origin, destination in od_pairs
    }
    metadata = {
        "strategy": {
            "candidates": strategy,
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
                "detour_rank": candidate_metadata[node].get("detour_rank"),
                "detour_rank_fraction": candidate_metadata[node].get(
                    "detour_rank_fraction"
                ),
                "minimum_detour_ratio": candidate_metadata[node].get(
                    "minimum_detour_ratio"
                ),
                "closest_od": candidate_metadata[node].get("closest_od"),
                "detour_ratios": candidate_metadata[node].get("detour_ratios"),
                "minimum_corridor_distance_m": candidate_metadata[node].get(
                    "minimum_corridor_distance_m"
                ),
                "corridor_distances_m": candidate_metadata[node].get(
                    "corridor_distances_m"
                ),
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
    metadata["saturation_diagnostics"] = saturation_diagnostics(
        graph, od_pairs, candidates, demand, calibration_window_hours
    )
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
