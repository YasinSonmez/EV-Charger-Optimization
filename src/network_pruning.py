"""Topology-preserving road-network pruning and diagnostics.

The functions in this module are intentionally independent of charger demand
and optimization.  They operate on OSMnx ``MultiDiGraph`` objects and retain
enough provenance to audit every topological simplification.
"""

from __future__ import annotations

import copy
import json
import math
import pickle
import random
import re
import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import osmnx as ox
from pyproj import CRS, Transformer
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial import cKDTree
from shapely.geometry import LineString
from shapely.ops import transform as transform_geometry


ROAD_PROFILES = {
    "primary_plus": (
        "motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link",
    ),
    "secondary_plus": (
        "motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link", "secondary", "secondary_link",
    ),
    "tertiary_plus": (
        "motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link", "secondary", "secondary_link",
        "tertiary", "tertiary_link",
    ),
}


def crs_is_projected(crs) -> bool:
    return bool(CRS.from_user_input(crs).is_projected)


def project_graph(graph: nx.MultiDiGraph, to_crs=None) -> nx.MultiDiGraph:
    """Project graph nodes/geometries without GeoPandas-version coupling."""
    source_crs = CRS.from_user_input(graph.graph.get("crs", "EPSG:4326"))
    if to_crs is None:
        if source_crs.is_projected:
            return copy.deepcopy(graph)
        xs = [float(data["x"]) for _, data in graph.nodes(data=True)]
        ys = [float(data["y"]) for _, data in graph.nodes(data=True)]
        mean_lon = float(np.mean(xs))
        mean_lat = float(np.mean(ys))
        zone = max(1, min(60, int((mean_lon + 180.0) // 6.0) + 1))
        epsg = (32600 if mean_lat >= 0 else 32700) + zone
        target_crs = CRS.from_epsg(epsg)
    else:
        target_crs = CRS.from_user_input(to_crs)
    if source_crs == target_crs:
        return copy.deepcopy(graph)
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    result = copy.deepcopy(graph)
    for _, data in result.nodes(data=True):
        data["x"], data["y"] = transformer.transform(float(data["x"]), float(data["y"]))
    for _, _, _, data in result.edges(keys=True, data=True):
        geometry = data.get("geometry")
        if geometry is not None:
            data["geometry"] = transform_geometry(transformer.transform, geometry)
    result.graph["crs"] = target_crs
    return result

PROFILE_ORDER = ("primary_plus", "secondary_plus", "tertiary_plus")
DEFAULT_SPEED_KPH = {
    "motorway": 105.0,
    "motorway_link": 65.0,
    "trunk": 90.0,
    "trunk_link": 55.0,
    "primary": 70.0,
    "primary_link": 45.0,
    "secondary": 55.0,
    "secondary_link": 40.0,
    "tertiary": 45.0,
    "tertiary_link": 35.0,
}


def highway_values(value) -> tuple[str, ...]:
    """Return normalized OSM highway tags without discarding list members."""
    if value is None:
        return ()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        values = []
        for item in value:
            values.extend(highway_values(item))
        return tuple(dict.fromkeys(values))
    return (str(value).strip(),)


def _flatten_unique(values) -> tuple[str, ...]:
    flattened = []
    for value in values:
        if isinstance(value, (list, tuple, set, np.ndarray)):
            flattened.extend(str(item) for item in value)
        else:
            flattened.append(str(value))
    return tuple(sorted(set(flattened)))


def _make_hashable(value):
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return tuple(_make_hashable(item) for item in value)
    if isinstance(value, dict):
        return tuple(sorted((str(key), _make_hashable(item)) for key, item in value.items()))
    return value


def _parse_number(value, default: float) -> float:
    values = value if isinstance(value, (list, tuple, set, np.ndarray)) else [value]
    parsed = []
    for item in values:
        if item is None:
            continue
        match = re.search(r"[-+]?\d*\.?\d+", str(item))
        if match:
            parsed.append(float(match.group()))
    return min(parsed) if parsed else float(default)


def parse_lanes(value) -> float:
    return max(1.0, _parse_number(value, 1.0))


def parse_speed_kph(value, highway=None) -> float:
    tags = highway_values(highway)
    fallback = min((DEFAULT_SPEED_KPH[tag] for tag in tags if tag in DEFAULT_SPEED_KPH),
                   default=40.0)
    values = value if isinstance(value, (list, tuple, set, np.ndarray)) else [value]
    parsed = []
    for item in values:
        if item is None:
            continue
        match = re.search(r"[-+]?\d*\.?\d+", str(item))
        if not match:
            continue
        speed = float(match.group())
        if "mph" in str(item).lower():
            speed *= 1.609344
        parsed.append(speed)
    return max(1.0, min(parsed) if parsed else fallback)


def prepare_source_graph(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Copy a raw graph and attach stable travel-time/provenance attributes."""
    result = copy.deepcopy(graph)
    result.graph.pop("simplified", None)
    result.graph.pop("consolidated", None)
    for u, v, key, data in result.edges(keys=True, data=True):
        for attribute in ("highway", "maxspeed", "lanes", "name", "ref", "osmid"):
            if isinstance(data.get(attribute), list):
                data[attribute] = tuple(data[attribute])
        if data.get("geometry") is None:
            data["geometry"] = LineString([
                (result.nodes[u]["x"], result.nodes[u]["y"]),
                (result.nodes[v]["x"], result.nodes[v]["y"]),
            ])
        length = float(data.get("length", 0.0) or 0.0)
        if not math.isfinite(length) or length <= 0:
            geometry = data.get("geometry")
            if geometry is not None and result.graph.get("crs") and not crs_is_projected(result.graph["crs"]):
                coords = list(geometry.coords)
                length = sum(
                    ox.distance.great_circle(lat1=y1, lon1=x1, lat2=y2, lon2=x2)
                    for (x1, y1), (x2, y2) in zip(coords[:-1], coords[1:])
                )
            elif geometry is not None:
                length = float(geometry.length)
        speed_kph = parse_speed_kph(data.get("maxspeed"), data.get("highway"))
        data["length"] = float(length)
        data["speed_kph"] = speed_kph
        data["travel_time"] = length / (speed_kph * 1000.0 / 3600.0) if length > 0 else math.nan
        data["lanes_numeric"] = parse_lanes(data.get("lanes"))
        data["source_edge_ids"] = (f"{u}|{v}|{key}",)
    return result


def filter_highways(graph: nx.MultiDiGraph, highway_types) -> nx.MultiDiGraph:
    """Filter locally while correctly retaining any matching mixed tag."""
    allowed = set(highway_types)
    result = copy.deepcopy(graph)
    remove = [
        (u, v, key)
        for u, v, key, data in result.edges(keys=True, data=True)
        if not (set(highway_values(data.get("highway"))) & allowed)
    ]
    result.remove_edges_from(remove)
    result.remove_nodes_from(list(nx.isolates(result)))
    result.graph.pop("simplified", None)
    result.graph.pop("consolidated", None)
    return result


def topology_simplify(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Remove only nondecision topology nodes and preserve path attributes."""
    current = copy.deepcopy(graph)
    for _ in range(10):
        current.graph.pop("simplified", None)
        current.graph.pop("consolidated", None)
        for _, _, _, data in current.edges(keys=True, data=True):
            for attribute, value in list(data.items()):
                if isinstance(value, (list, tuple, set, np.ndarray, dict)):
                    data[attribute] = _make_hashable(value)
        # OSMnx intentionally keeps only one edge when a simplification path
        # traverses parallel edges. Such nodes are routing/multigraph
        # structure, so force them to remain endpoints.
        for u, v in current.edges(keys=False):
            if current.number_of_edges(u, v) > 1:
                current.nodes[u]["_parallel_endpoint"] = True
                current.nodes[v]["_parallel_endpoint"] = True
        prior_nodes = len(current)
        simplified = ox.simplification.simplify_graph(
            current,
            node_attrs_include=["_parallel_endpoint", "_consolidated_endpoint"],
            remove_rings=False,
            track_merged=False,
            edge_attr_aggs={
                "length": sum,
                "travel_time": sum,
                "source_edge_ids": _flatten_unique,
                "lanes_numeric": min,
            },
        )
        for _, _, _, data in simplified.edges(keys=True, data=True):
            provenance = data.get("source_edge_ids", ())
            data["source_edge_ids"] = _flatten_unique([provenance])
            length = float(data.get("length", 0.0) or 0.0)
            travel_time = float(data.get("travel_time", 0.0) or 0.0)
            if travel_time > 0 and math.isfinite(travel_time):
                data["speed_kph"] = length / travel_time * 3.6
            else:
                speed = parse_speed_kph(data.get("maxspeed"), data.get("highway"))
                data["speed_kph"] = speed
                data["travel_time"] = length / (speed * 1000.0 / 3600.0)
            data["lanes_numeric"] = parse_lanes(data.get("lanes_numeric", data.get("lanes")))
        current = simplified
        if not removable_pass_through_nodes(current) or len(current) == prior_nodes:
            break
    return current


def _original_nodes(node_data, node_id) -> tuple:
    originals = node_data.get("osmid_original", node_id)
    if isinstance(originals, (list, tuple, set, np.ndarray)):
        return tuple(originals)
    return (originals,)


def consolidate_intersections(
    projected_topology: nx.MultiDiGraph,
    radius_m: float,
    *,
    require_induced_strong_connectivity: bool = False,
) -> tuple[nx.MultiDiGraph, dict, dict]:
    """Consolidate connected nearby nodes and return auditable node mappings."""
    radius_m = float(radius_m)
    if radius_m < 0:
        raise ValueError("intersection consolidation radius must be non-negative")
    if radius_m == 0:
        result = copy.deepcopy(projected_topology)
        mapping = {node: node for node in result.nodes}
        cluster_info = {
            node: {"size": 1, "diameter_m": 0.0, "absorbed_source_edge_ids": ()}
            for node in result.nodes
        }
        for node in result.nodes:
            result.nodes[node]["consolidated_count"] = 1
            result.nodes[node]["cluster_diameter_m"] = 0.0
            result.nodes[node]["absorbed_source_edge_ids"] = ()
        return result, mapping, cluster_info

    source = copy.deepcopy(projected_topology)
    source.graph.pop("consolidated", None)
    undirected = nx.Graph()
    undirected.add_nodes_from(source.nodes)
    undirected.add_edges_from((u, v) for u, v in source.edges(keys=False) if u != v)
    eligible = [
        node for node in sorted(source.nodes, key=str)
        if len(set(undirected.neighbors(node))) > 1
    ]
    points = np.asarray([[source.nodes[node]["x"], source.nodes[node]["y"]] for node in eligible])
    proximity = nx.Graph()
    proximity.add_nodes_from(eligible)
    if len(points) > 1:
        tree = cKDTree(points)
        for left, right in sorted(tree.query_pairs(2.0 * radius_m)):
            u, v = eligible[left], eligible[right]
            # Geographic proximity is not sufficient: merging is allowed only
            # when the nodes are directly joined in the road topology.
            if undirected.has_edge(u, v):
                proximity.add_edge(u, v)

    clusters = []
    for component in nx.connected_components(proximity):
        members = sorted(component, key=str)
        if len(members) < 2:
            continue
        member_points = np.asarray([[source.nodes[node]["x"], source.nodes[node]["y"]] for node in members])
        labels = fcluster(
            linkage(member_points, method="complete"),
            t=2.0 * radius_m,
            criterion="distance",
        )
        by_label = defaultdict(list)
        for node, label in zip(members, labels):
            by_label[int(label)].append(node)
        for label in sorted(by_label):
            candidate = by_label[label]
            # Complete-linkage controls diameter; this second split guarantees
            # every final cluster is also topologically connected.
            for connected in nx.connected_components(undirected.subgraph(candidate)):
                connected = sorted(connected, key=str)
                groups = [connected]
                if require_induced_strong_connectivity:
                    groups = [
                        sorted(component, key=str)
                        for component in nx.strongly_connected_components(
                            source.subgraph(connected)
                        )
                    ]
                clusters.extend(group for group in groups if len(group) > 1)

    cluster_for = {}
    cluster_members = {}
    for index, members in enumerate(sorted(clusters, key=lambda values: tuple(map(str, values)))):
        cluster_id = f"_J{index}"
        cluster_members[cluster_id] = tuple(members)
        for node in members:
            cluster_for[node] = cluster_id
    mapping = {node: cluster_for.get(node, node) for node in source.nodes}

    result = nx.MultiDiGraph()
    result.graph = copy.deepcopy(source.graph)
    result.graph["consolidated"] = True
    for node, data in source.nodes(data=True):
        if node in cluster_for:
            continue
        copied = copy.deepcopy(data)
        copied["osmid_original"] = _original_nodes(copied, node)
        copied["consolidated_count"] = 1
        copied["cluster_diameter_m"] = 0.0
        copied["absorbed_source_edge_ids"] = ()
        result.add_node(node, **copied)

    cluster_info = {}
    for cluster_id, members in cluster_members.items():
        cluster_points = np.asarray([[source.nodes[node]["x"], source.nodes[node]["y"]] for node in members])
        centroid = np.mean(cluster_points, axis=0)
        diameter = float(max(
            np.linalg.norm(cluster_points[i] - cluster_points[j])
            for i in range(len(cluster_points)) for j in range(i + 1, len(cluster_points))
        ))
        external_neighbors = {
            neighbor for node in members for neighbor in undirected.neighbors(node)
            if neighbor not in set(members)
        }
        result.add_node(
            cluster_id,
            x=float(centroid[0]),
            y=float(centroid[1]),
            osmid_original=members,
            consolidated_count=len(members),
            cluster_diameter_m=diameter,
            external_branch_count=len(external_neighbors),
            absorbed_source_edge_ids=(),
            _consolidated_endpoint=True,
        )
        cluster_info[cluster_id] = {
            "size": len(members),
            "diameter_m": diameter,
            "external_branch_count": len(external_neighbors),
            "absorbed_source_edge_ids": (),
        }

    absorbed_by_cluster = defaultdict(list)
    for u, v, key, edge_data in source.edges(keys=True, data=True):
        new_u, new_v = mapping[u], mapping[v]
        if new_u == new_v and u != v:
            absorbed_by_cluster[new_u].extend(edge_data.get("source_edge_ids", ()))
            continue
        data = copy.deepcopy(edge_data)
        data["source_edge_ids"] = _flatten_unique([data.get("source_edge_ids", ())])
        geometry = data.get("geometry")
        if geometry is not None and hasattr(geometry, "coords"):
            coords = list(geometry.coords)
            if new_u != u:
                coords[0] = (result.nodes[new_u]["x"], result.nodes[new_u]["y"])
            if new_v != v:
                coords[-1] = (result.nodes[new_v]["x"], result.nodes[new_v]["y"])
            if len(coords) >= 2:
                data["geometry"] = LineString(coords)
                data["length"] = float(data["geometry"].length)
        length = float(data.get("length", 0.0) or 0.0)
        speed = float(data.get("speed_kph", 0.0) or 0.0)
        if speed <= 0 or not math.isfinite(speed):
            speed = parse_speed_kph(data.get("maxspeed"), data.get("highway"))
        data["speed_kph"] = speed
        data["travel_time"] = length / (speed * 1000.0 / 3600.0) if length > 0 else math.nan
        data["lanes_numeric"] = parse_lanes(data.get("lanes_numeric", data.get("lanes")))
        data["source_edge_key"] = str(key)
        result.add_edge(new_u, new_v, **data)

    for cluster_id, absorbed in absorbed_by_cluster.items():
        values = tuple(sorted(set(str(value) for value in absorbed)))
        result.nodes[cluster_id]["absorbed_source_edge_ids"] = values
        cluster_info[cluster_id]["absorbed_source_edge_ids"] = values
    # Consolidation can turn nodes adjacent to a merged cluster into new
    # pass-throughs. Remove those to a fixed point while protecting the
    # consolidated intersection itself.
    result = topology_simplify(result)
    mapping = {
        original: target if target in result else None
        for original, target in mapping.items()
    }
    return result, mapping, cluster_info


def largest_component(graph: nx.MultiDiGraph, strong: bool) -> nx.MultiDiGraph:
    components = (
        nx.strongly_connected_components(graph)
        if strong else nx.weakly_connected_components(graph)
    )
    components = list(components)
    if not components:
        return graph.__class__()
    nodes = max(components, key=lambda component: (len(component), tuple(sorted(map(str, component)))))
    return graph.subgraph(nodes).copy()


def removable_pass_through_nodes(graph: nx.MultiDiGraph) -> list:
    """Return nodes matching OSMnx's strict removable topology rule."""
    removable = []
    for node, data in graph.nodes(data=True):
        if int(data.get("consolidated_count", 1)) > 1 or data.get("_parallel_endpoint"):
            continue
        neighbors = set(graph.predecessors(node)) | set(graph.successors(node))
        if node in neighbors or graph.in_degree(node) == 0 or graph.out_degree(node) == 0:
            continue
        if len(neighbors) == 2 and graph.degree(node) in {2, 4}:
            removable.append(node)
    return sorted(removable, key=str)


def _sample_pairs(nodes, seed=42, sources=20, targets_per_source=10):
    ordered = sorted(nodes, key=str)
    if len(ordered) < 2:
        return []
    rng = random.Random(seed)
    selected_sources = rng.sample(ordered, min(sources, len(ordered)))
    pairs = []
    for source in selected_sources:
        candidates = [node for node in ordered if node != source]
        for target in rng.sample(candidates, min(targets_per_source, len(candidates))):
            pairs.append((source, target))
    return pairs


def path_fidelity(reference: nx.MultiDiGraph, candidate: nx.MultiDiGraph,
                  mapping: dict, seed=42) -> dict:
    """Measure sampled reachability changes and shortest-time distortion."""
    mapped_nodes = [node for node in reference if mapping.get(node) in candidate]
    pairs = _sample_pairs(mapped_nodes, seed=seed)
    by_source = defaultdict(list)
    for source, target in pairs:
        by_source[source].append(target)

    compared = 0
    reachable_before = 0
    preserved = 0
    introduced = 0
    distortions = []
    for source, targets in by_source.items():
        ref_distances = nx.single_source_dijkstra_path_length(
            reference, source, weight="travel_time"
        )
        mapped_source = mapping.get(source)
        cand_distances = {}
        if mapped_source in candidate:
            cand_distances = nx.single_source_dijkstra_path_length(
                candidate, mapped_source, weight="travel_time"
            )
        for target in targets:
            compared += 1
            mapped_target = mapping.get(target)
            before = ref_distances.get(target)
            after = cand_distances.get(mapped_target) if mapped_target is not None else None
            if before is None:
                if after is not None and mapped_source != mapped_target:
                    introduced += 1
                continue
            reachable_before += 1
            if after is None:
                continue
            preserved += 1
            if before > 0 and mapped_source != mapped_target:
                distortions.append(abs(float(after) - float(before)) / float(before))

    values = np.asarray(distortions, dtype=float)
    return {
        "mapping_coverage": len(mapped_nodes) / len(reference) if len(reference) else 1.0,
        "sample_pairs": compared,
        "reachable_before": reachable_before,
        "reachability_preserved": preserved,
        "reachability_preserved_fraction": (
            preserved / reachable_before if reachable_before else 1.0
        ),
        "introduced_reachable_pairs": introduced,
        "path_distortion_median": float(np.median(values)) if len(values) else 0.0,
        "path_distortion_p95": float(np.percentile(values, 95)) if len(values) else 0.0,
        "path_distortion_max": float(np.max(values)) if len(values) else 0.0,
    }


def _close_node_metrics(graph: nx.MultiDiGraph) -> dict:
    if not graph.nodes:
        return {
            "nearest_p05_m": math.nan,
            "nearest_median_m": math.nan,
            "nearest_p95_m": math.nan,
            "close_pairs": {str(value): 0 for value in (10, 20, 30, 50)},
            "unconnected_close_pairs": {str(value): 0 for value in (10, 20, 30, 50)},
            "nearest_distances": [],
        }
    nodes = list(graph.nodes)
    points = np.array([[graph.nodes[node]["x"], graph.nodes[node]["y"]] for node in nodes])
    tree = cKDTree(points)
    distances, _ = tree.query(points, k=min(2, len(points)))
    nearest = np.zeros(len(points)) if len(points) == 1 else distances[:, 1]
    close = {}
    unconnected = {}
    for threshold in (10, 20, 30, 50):
        pairs = tree.query_pairs(threshold)
        close[str(threshold)] = len(pairs)
        unconnected[str(threshold)] = sum(
            1 for left, right in pairs
            if not graph.has_edge(nodes[left], nodes[right])
            and not graph.has_edge(nodes[right], nodes[left])
        )
    return {
        "nearest_p05_m": float(np.percentile(nearest, 5)),
        "nearest_median_m": float(np.median(nearest)),
        "nearest_p95_m": float(np.percentile(nearest, 95)),
        "close_pairs": close,
        "unconnected_close_pairs": unconnected,
        "nearest_distances": nearest.tolist(),
    }


def graph_metrics(graph: nx.MultiDiGraph, *, area_km2: float,
                  profile: str, radius_m: float, connector_recovery: bool,
                  fidelity: dict, elapsed_seconds: float) -> dict:
    """Compute all scalar diagnostics used to rank pruning variants."""
    wcc = largest_component(graph, strong=False)
    scc = largest_component(graph, strong=True)
    weak_components = nx.number_weakly_connected_components(graph) if graph else 0
    strong_components = nx.number_strongly_connected_components(graph) if graph else 0
    road_lengths = Counter()
    invalid_edges = 0
    missing_provenance = 0
    total_length = 0.0
    for _, _, _, data in graph.edges(keys=True, data=True):
        length = float(data.get("length", math.nan))
        travel_time = float(data.get("travel_time", math.nan))
        if not math.isfinite(length) or length <= 0 or not math.isfinite(travel_time) or travel_time <= 0:
            invalid_edges += 1
        total_length += length if math.isfinite(length) else 0.0
        tags = highway_values(data.get("highway")) or ("unknown",)
        share = length / len(tags) if tags else length
        for tag in tags:
            road_lengths[tag] += share
        if not data.get("source_edge_ids"):
            missing_provenance += 1
    cluster_records = [
        (
            str(node),
            int(data.get("consolidated_count", 1)),
            float(data.get("cluster_diameter_m", 0.0)),
            int(data.get("external_branch_count", 0)),
        )
        for node, data in graph.nodes(data=True)
        if int(data.get("consolidated_count", 1)) > 1
    ]
    cluster_records.sort(key=lambda item: item[0])
    cluster_sizes = [item[1] for item in cluster_records]
    cluster_diameters = [item[2] for item in cluster_records]
    cluster_external_branches = [item[3] for item in cluster_records]
    close = _close_node_metrics(graph)
    scc_ratio = len(scc) / len(wcc) if len(wcc) else 0.0
    metrics = {
        "profile": profile,
        "radius_m": float(radius_m),
        "connector_recovery": bool(connector_recovery),
        "nodes": len(graph),
        "edges": graph.number_of_edges(),
        "weak_components": weak_components,
        "strong_components": strong_components,
        "largest_wcc_nodes": len(wcc),
        "largest_scc_nodes": len(scc),
        "scc_wcc_ratio": scc_ratio,
        "node_density_per_km2": len(graph) / area_km2 if area_km2 > 0 else math.nan,
        "total_road_km": total_length / 1000.0,
        "road_km_by_class": json.dumps(
            {key: value / 1000.0 for key, value in sorted(road_lengths.items())},
            sort_keys=True,
        ),
        "removable_pass_through_nodes": len(removable_pass_through_nodes(graph)),
        "invalid_edges": invalid_edges,
        "missing_provenance_edges": missing_provenance,
        "consolidation_clusters": len(cluster_sizes),
        "largest_cluster_size": max(cluster_sizes, default=1),
        "largest_cluster_diameter_m": max(cluster_diameters, default=0.0),
        "cluster_size_median": float(np.median(cluster_sizes)) if cluster_sizes else 1.0,
        "cluster_size_p95": float(np.percentile(cluster_sizes, 95)) if cluster_sizes else 1.0,
        "cluster_diameter_median_m": (
            float(np.median(cluster_diameters)) if cluster_diameters else 0.0
        ),
        "cluster_diameter_p95_m": (
            float(np.percentile(cluster_diameters, 95)) if cluster_diameters else 0.0
        ),
        "cluster_external_branches_median": (
            float(np.median(cluster_external_branches)) if cluster_external_branches else 0.0
        ),
        "cluster_external_branches_p95": (
            float(np.percentile(cluster_external_branches, 95)) if cluster_external_branches else 0.0
        ),
        "cluster_external_branches_max": max(cluster_external_branches, default=0),
        "cluster_sizes": json.dumps(cluster_sizes),
        "cluster_diameters_m": json.dumps(cluster_diameters),
        "cluster_external_branch_counts": json.dumps(cluster_external_branches),
        "nearest_p05_m": close["nearest_p05_m"],
        "nearest_median_m": close["nearest_median_m"],
        "nearest_p95_m": close["nearest_p95_m"],
        "serialized_bytes": len(pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL)),
        "elapsed_seconds": float(elapsed_seconds),
        **fidelity,
    }
    for threshold in (10, 20, 30, 50):
        metrics[f"close_pairs_le_{threshold}m"] = close["close_pairs"][str(threshold)]
        metrics[f"unconnected_close_pairs_le_{threshold}m"] = close["unconnected_close_pairs"][str(threshold)]
        metrics[f"adjacent_close_pairs_le_{threshold}m"] = (
            close["close_pairs"][str(threshold)]
            - close["unconnected_close_pairs"][str(threshold)]
        )
    metrics["eligible"] = bool(
        metrics["removable_pass_through_nodes"] == 0
        and invalid_edges == 0
        and missing_provenance == 0
        and metrics["reachability_preserved_fraction"] >= 1.0
        and metrics["introduced_reachable_pairs"] == 0
        and metrics["path_distortion_median"] <= 0.02
        and metrics["path_distortion_p95"] <= 0.05
        and scc_ratio >= 0.90
    )
    return metrics


def _copy_path_edges(destination: nx.MultiDiGraph, source: nx.MultiDiGraph, path) -> int:
    added = 0
    for node in path:
        if node not in destination:
            destination.add_node(node, **copy.deepcopy(source.nodes[node]))
    for u, v in zip(path[:-1], path[1:]):
        candidates = source.get_edge_data(u, v) or {}
        if not candidates:
            continue
        _, data = min(
            candidates.items(),
            key=lambda item: (float(item[1].get("travel_time", math.inf)), str(item[0])),
        )
        source_ids = set(data.get("source_edge_ids", ()))
        already_present = any(
            source_ids and source_ids.issubset(set(existing.get("source_edge_ids", ())))
            for existing in (destination.get_edge_data(u, v) or {}).values()
        )
        if already_present:
            continue
        destination.add_edge(u, v, **copy.deepcopy(data))
        added += 1
    return added


def recover_directed_connectors(narrow: nx.MultiDiGraph, broad: nx.MultiDiGraph,
                                target_ratio=0.90, max_rounds=20):
    """Add minimum-time broader-profile paths to improve SCC/WCC retention."""
    recovered = copy.deepcopy(narrow)
    metadata = {"attempted": True, "paths_added": 0, "edges_added": 0, "rounds": 0}
    for round_index in range(int(max_rounds)):
        wcc = largest_component(recovered, strong=False)
        if not wcc:
            break
        sccs = sorted(
            nx.strongly_connected_components(wcc),
            key=lambda component: (-len(component), tuple(sorted(map(str, component)))),
        )
        main = set(sccs[0])
        ratio = len(main) / len(wcc)
        if ratio >= float(target_ratio) or len(sccs) == 1:
            break
        target_component = set(sccs[1])
        broad_nodes = set(broad)
        main &= broad_nodes
        target_component &= broad_nodes
        if not main or not target_component:
            break

        distances_out, paths_out = nx.multi_source_dijkstra(
            broad, main, weight="travel_time"
        )
        reachable_out = [node for node in target_component if node in distances_out]
        reverse = broad.reverse(copy=False)
        distances_back, paths_back = nx.multi_source_dijkstra(
            reverse, main, weight="travel_time"
        )
        reachable_back = [node for node in target_component if node in distances_back]
        if not reachable_out or not reachable_back:
            break
        out_target = min(reachable_out, key=lambda node: (distances_out[node], str(node)))
        back_target = min(reachable_back, key=lambda node: (distances_back[node], str(node)))
        paths = [paths_out[out_target], list(reversed(paths_back[back_target]))]
        before_edges = recovered.number_of_edges()
        for path in paths:
            metadata["edges_added"] += _copy_path_edges(recovered, broad, path)
            metadata["paths_added"] += 1
        metadata["rounds"] = round_index + 1
        if recovered.number_of_edges() == before_edges:
            break
    final_wcc = largest_component(recovered, strong=False)
    final_scc = largest_component(recovered, strong=True)
    metadata["final_scc_wcc_ratio"] = len(final_scc) / len(final_wcc) if len(final_wcc) else 0.0
    return recovered, metadata


def build_profile(source_graph: nx.MultiDiGraph, profile: str,
                  connector_source: nx.MultiDiGraph | None = None,
                  connector_threshold=0.90):
    """Filter and simplify one road profile, with conditional recovery."""
    if profile not in ROAD_PROFILES:
        raise ValueError(f"unknown road profile: {profile}")
    filtered = filter_highways(source_graph, ROAD_PROFILES[profile])
    topology = topology_simplify(filtered)
    projected = project_graph(topology)
    wcc = largest_component(projected, strong=False)
    scc = largest_component(projected, strong=True)
    ratio = len(scc) / len(wcc) if len(wcc) else 0.0
    connector_metadata = {"attempted": False, "final_scc_wcc_ratio": ratio}
    recovered_filtered = None
    recovered_topology = None
    recovered_projected = None
    if ratio < float(connector_threshold) and connector_source is not None:
        recovered_filtered, connector_metadata = recover_directed_connectors(
            filtered, connector_source, target_ratio=connector_threshold
        )
        if (
            connector_metadata.get("edges_added", 0) > 0
            and connector_metadata.get("final_scc_wcc_ratio", 0.0) > ratio + 1e-12
        ):
            recovered_topology = topology_simplify(recovered_filtered)
            recovered_projected = project_graph(recovered_topology)
        else:
            recovered_filtered = None
    return {
        "filtered": filtered,
        "topology": topology,
        "projected": projected,
        "recovered_filtered": recovered_filtered,
        "recovered_topology": recovered_topology,
        "recovered_projected": recovered_projected,
        "connector_metadata": connector_metadata,
    }


def timed_consolidation(reference, radius_m, seed=42, *,
                        require_induced_strong_connectivity=False):
    started = time.perf_counter()
    graph, mapping, clusters = consolidate_intersections(
        reference,
        radius_m,
        require_induced_strong_connectivity=require_induced_strong_connectivity,
    )
    fidelity = path_fidelity(reference, graph, mapping, seed=seed)
    return graph, mapping, clusters, fidelity, time.perf_counter() - started
