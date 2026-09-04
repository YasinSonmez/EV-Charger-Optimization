"""Diagnostic coarse representations for validated road-network artifacts.

The candidate layer is safe for charger-site selection because routing remains
on the original graph.  The shortcut skeleton preserves free-flow distances but
must not be used as an independent-capacity congestion network.
"""

from __future__ import annotations

import copy
import heapq
import math
from collections import defaultdict

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree

from src.network_pruning import highway_values


def _points(graph, nodes):
    return np.asarray([
        [float(graph.nodes[node]["x"]), float(graph.nodes[node]["y"])]
        for node in nodes
    ])


def _medoid_near_centroid(graph, nodes):
    ordered = sorted(nodes, key=str)
    points = _points(graph, ordered)
    center = points.mean(axis=0)
    index = min(
        range(len(ordered)),
        key=lambda i: (float(np.sum((points[i] - center) ** 2)), str(ordered[i])),
    )
    return ordered[index]


def select_charger_candidate_layer(
    graph: nx.MultiDiGraph,
    *,
    max_candidates: int = 1000,
    interchange_merge_diameter_m: float = 250.0,
):
    """Select ≤``max_candidates`` spatially covering road decision sites.

    Link-road components seed the selection. Nearby component representatives
    are grouped with complete linkage so one interchange does not consume many
    candidates. Remaining slots use deterministic farthest-point sampling over
    the routing nodes, preserving coverage of long corridors and sparse areas.
    """
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    if interchange_merge_diameter_m <= 0:
        raise ValueError("interchange_merge_diameter_m must be positive")
    nodes = sorted(graph.nodes, key=str)
    if not nodes:
        return [], {}, {"candidate_count": 0}

    link_graph = nx.Graph()
    for u, v, data in graph.edges(data=True):
        if any(tag.endswith("_link") for tag in highway_values(data.get("highway"))):
            link_graph.add_edge(u, v)
    raw_components = [
        sorted(component, key=str)
        for component in nx.connected_components(link_graph)
        if len(component) > 1
    ]
    raw_representatives = [
        _medoid_near_centroid(graph, component) for component in raw_components
    ]

    interchange_groups = []
    if len(raw_representatives) == 1:
        interchange_groups = [raw_representatives]
    elif len(raw_representatives) > 1:
        labels = fcluster(
            linkage(_points(graph, raw_representatives), method="complete"),
            t=float(interchange_merge_diameter_m),
            criterion="distance",
        )
        by_label = defaultdict(list)
        for node, label in zip(raw_representatives, labels):
            by_label[int(label)].append(node)
        interchange_groups = [by_label[label] for label in sorted(by_label)]

    target = min(int(max_candidates), len(nodes))
    selected = []
    metadata = {}
    for group in interchange_groups:
        representative = _medoid_near_centroid(graph, group)
        if representative in metadata:
            continue
        selected.append(representative)
        metadata[representative] = {
            "kind": "interchange",
            "component_representatives": tuple(sorted(group, key=str)),
        }

    if len(selected) > target:
        pool = sorted(selected, key=str)
        pool_points = _points(graph, pool)
        first = min(
            range(len(pool)),
            key=lambda i: (
                float(pool_points[i, 0]), float(pool_points[i, 1]), str(pool[i])
            ),
        )
        retained_indices = [first]
        retained_mask = np.zeros(len(pool), dtype=bool)
        retained_mask[first] = True
        minimum_distance = np.linalg.norm(pool_points - pool_points[first], axis=1)
        while len(retained_indices) < target:
            scores = np.where(retained_mask, -1.0, minimum_distance)
            index = int(np.argmax(scores))
            retained_indices.append(index)
            retained_mask[index] = True
            minimum_distance = np.minimum(
                minimum_distance,
                np.linalg.norm(pool_points - pool_points[index], axis=1),
            )
        selected = [pool[index] for index in retained_indices]
        metadata = {node: metadata[node] for node in selected}

    points = _points(graph, nodes)
    node_index = {node: index for index, node in enumerate(nodes)}
    selected_mask = np.zeros(len(nodes), dtype=bool)
    if selected:
        selected_mask[[node_index[node] for node in selected]] = True
        minimum_distance = cKDTree(points[selected_mask]).query(points, k=1)[0]
    else:
        first = min(
            range(len(nodes)),
            key=lambda i: (float(points[i, 0]), float(points[i, 1]), str(nodes[i])),
        )
        selected.append(nodes[first])
        selected_mask[first] = True
        metadata[nodes[first]] = {"kind": "coverage", "component_representatives": ()}
        minimum_distance = np.linalg.norm(points - points[first], axis=1)

    while len(selected) < target:
        scores = np.where(selected_mask, -1.0, minimum_distance)
        index = int(np.argmax(scores))
        node = nodes[index]
        selected.append(node)
        selected_mask[index] = True
        metadata[node] = {"kind": "coverage", "component_representatives": ()}
        minimum_distance = np.minimum(
            minimum_distance,
            np.linalg.norm(points - points[index], axis=1),
        )
        minimum_distance[index] = 0.0

    coverage = cKDTree(points[selected_mask]).query(points, k=1)[0]
    selected_points = points[selected_mask]
    selected_tree = cKDTree(selected_points)
    if len(selected_points) > 1:
        candidate_nearest = selected_tree.query(selected_points, k=2)[0][:, 1]
    else:
        candidate_nearest = np.asarray([0.0])
    diagnostics = {
        "candidate_count": len(selected),
        "raw_link_components": len(raw_components),
        "interchange_groups": len(interchange_groups),
        "interchange_candidates": sum(
            1 for node in selected if metadata[node]["kind"] == "interchange"
        ),
        "coverage_candidates": sum(
            1 for node in selected if metadata[node]["kind"] == "coverage"
        ),
        "network_node_distance_median_m": float(np.median(coverage)),
        "network_node_distance_p95_m": float(np.percentile(coverage, 95)),
        "network_node_distance_max_m": float(np.max(coverage)),
        "candidate_nearest_min_m": float(np.min(candidate_nearest)),
        "candidate_nearest_median_m": float(np.median(candidate_nearest)),
        "candidate_nearest_p95_m": float(np.percentile(candidate_nearest, 95)),
        "candidate_pairs_within_10m": len(selected_tree.query_pairs(10.0)),
        "candidate_pairs_within_20m": len(selected_tree.query_pairs(20.0)),
        "candidate_pairs_within_30m": len(selected_tree.query_pairs(30.0)),
        "candidate_pairs_within_50m": len(selected_tree.query_pairs(50.0)),
        "candidate_pairs_within_100m": len(selected_tree.query_pairs(100.0)),
    }
    return selected, metadata, diagnostics


def _minimum_edge(graph, u, v):
    edges = graph.get_edge_data(u, v) or {}
    if not edges:
        return None
    return min(
        edges.values(),
        key=lambda data: (float(data.get("travel_time", math.inf)),
                          tuple(map(str, data.get("source_edge_ids", ())))),
    )


def _contraction_priority(graph, node):
    predecessors = set(graph.predecessors(node)) - {node}
    successors = set(graph.successors(node)) - {node}
    return (len(predecessors) * len(successors), graph.degree(node), str(node))


def contract_freeflow_shortcut_skeleton(
    graph: nx.MultiDiGraph,
    *,
    target_nodes: int = 1000,
    edge_limit: int = 200_000,
):
    """Contract to a free-flow shortest-path skeleton with path provenance.

    This is diagnostic only. Shortcut edges can have overlapping provenance and
    therefore cannot be assigned independent BPR capacities or queue servers.
    """
    if target_nodes < 2:
        raise ValueError("target_nodes must be at least 2")
    result = copy.deepcopy(graph)
    if len(result) <= target_nodes:
        return result, {"nodes_removed": 0, "shortcuts_added": 0}

    versions = {node: 0 for node in result}
    heap = []
    for node in result:
        heapq.heappush(heap, (*_contraction_priority(result, node), 0, node))
    shortcuts_added = 0
    nodes_removed = 0

    while len(result) > target_nodes:
        if not heap:
            raise RuntimeError("contraction priority queue was exhausted")
        *_, version, node = heapq.heappop(heap)
        if node not in result or version != versions[node]:
            continue
        incoming = {
            predecessor: _minimum_edge(result, predecessor, node)
            for predecessor in set(result.predecessors(node)) - {node}
        }
        outgoing = {
            successor: _minimum_edge(result, node, successor)
            for successor in set(result.successors(node)) - {node}
        }
        touched = set(incoming) | set(outgoing)
        shortcuts = []
        for predecessor, incoming_data in incoming.items():
            for successor, outgoing_data in outgoing.items():
                if predecessor == successor:
                    continue
                travel_time = (
                    float(incoming_data["travel_time"])
                    + float(outgoing_data["travel_time"])
                )
                existing = _minimum_edge(result, predecessor, successor)
                if existing is not None and float(existing["travel_time"]) <= travel_time * (1 + 1e-12):
                    continue
                provenance = tuple(dict.fromkeys((
                    *incoming_data.get("source_edge_ids", ()),
                    *outgoing_data.get("source_edge_ids", ()),
                )))
                shortcuts.append((predecessor, successor, {
                    "travel_time": travel_time,
                    "length": float(incoming_data.get("length", 0.0))
                    + float(outgoing_data.get("length", 0.0)),
                    "source_edge_ids": provenance,
                    "shortcut": True,
                }))
        result.remove_node(node)
        nodes_removed += 1
        for predecessor, successor, data in shortcuts:
            result.add_edge(predecessor, successor, **data)
            shortcuts_added += 1
        if result.number_of_edges() > edge_limit:
            raise RuntimeError(
                f"shortcut skeleton exceeded edge_limit={edge_limit}"
            )
        for touched_node in touched:
            if touched_node in result:
                versions[touched_node] += 1
                heapq.heappush(
                    heap,
                    (*_contraction_priority(result, touched_node),
                     versions[touched_node], touched_node),
                )

    provenance_occurrences = sum(
        len(data.get("source_edge_ids", ()))
        for _, _, data in result.edges(data=True)
    )
    unique_provenance = {
        source_id
        for _, _, data in result.edges(data=True)
        for source_id in data.get("source_edge_ids", ())
    }
    diagnostics = {
        "nodes_removed": nodes_removed,
        "shortcuts_added": shortcuts_added,
        "final_nodes": len(result),
        "final_edges": result.number_of_edges(),
        "provenance_occurrences": provenance_occurrences,
        "unique_source_edges": len(unique_provenance),
        "provenance_overlap_ratio": (
            provenance_occurrences / len(unique_provenance)
            if unique_provenance else 0.0
        ),
        "max_source_edges_per_shortcut": max(
            (len(data.get("source_edge_ids", ()))
             for _, _, data in result.edges(data=True)),
            default=0,
        ),
        "congestion_compatible": False,
    }
    return result, diagnostics
