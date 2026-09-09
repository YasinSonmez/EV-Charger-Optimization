"""Deterministic queue route libraries and balanced charger quotas."""

from __future__ import annotations

from itertools import islice

import networkx as nx
import pandas as pd


def _edge_time(row) -> float:
    value = getattr(row, "travel_time", None)
    if value is not None and pd.notna(value) and float(value) > 0:
        return float(value)
    speed = max(float(getattr(row, "maxmph", 25.0) or 25.0), 1.0)
    return float(row.length) / (speed * 0.44704)


def _routing_graph(edges):
    """Collapse parallel links by deterministic minimum free-flow time."""
    graph = nx.DiGraph()
    for row in edges.sort_values("link_id", kind="mergesort").itertuples():
        u, v, link_id = int(row.start_node_id), int(row.end_node_id), int(row.link_id)
        travel_time = _edge_time(row)
        old = graph.get_edge_data(u, v)
        if old is None or (travel_time, link_id) < (old["travel_time"], old["link_id"]):
            graph.add_edge(u, v, travel_time=travel_time, link_id=link_id)
    return graph


def _path_record(graph, path, *, od, vehicle_type, charger=None, rank=0):
    link_ids = [int(graph[u][v]["link_id"]) for u, v in zip(path, path[1:])]
    travel_time = sum(float(graph[u][v]["travel_time"]) for u, v in zip(path, path[1:]))
    station = "none" if charger is None else str(int(charger))
    return {
        "route_id": f"{od[0]}_{od[1]}_{vehicle_type}_{station}_{rank}",
        "path": [int(node) for node in path],
        "link_ids": link_ids,
        "flow": 0.0,
        "station node": None if charger is None else int(charger),
        "free_flow_time": float(travel_time),
    }


def _shortest_paths(graph, origin, destination, limit):
    try:
        return list(islice(nx.shortest_simple_paths(
            graph, int(origin), int(destination), weight="travel_time"
        ), int(limit)))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def balanced_charger_routes(routes_by_charger, total_k, rank_key):
    """Select up to K F2 routes, keeping charger counts as even as supply allows.

    Slots are assigned one at a time to the least-loaded charger with a
    remaining candidate; ties use the lowest-ranked next route, then charger
    id. With sufficient supply each charger receives floor/ceil(K/C) routes.
    Chargers with fewer feasible routes are capped, and other chargers fill
    the remaining slots. Fails only when no charger has a feasible route.
    """
    chargers = sorted(int(value) for value in routes_by_charger)
    if not chargers:
        raise ValueError("Cannot allocate charging routes without installed chargers")
    total_k = int(total_k)
    ordered = {
        charger: sorted(routes_by_charger[charger], key=rank_key)
        for charger in chargers
    }
    if all(not routes for routes in ordered.values()):
        raise ValueError("No feasible charging routes through any installed charger")
    quotas = {charger: 0 for charger in chargers}
    allocated = 0
    while allocated < total_k:
        eligible = [
            charger for charger in chargers
            if len(ordered[charger]) > quotas[charger]
        ]
        if not eligible:
            break
        charger = min(
            eligible,
            key=lambda value: (
                quotas[value],
                rank_key(ordered[value][quotas[value]]),
                value,
            ),
        )
        quotas[charger] += 1
        allocated += 1
    return [route for charger in chargers for route in ordered[charger][:quotas[charger]]]


def independent_flow_data(edges, demand_classes, chargers, k, cache=None):
    """Build K F1 and K balanced F2 free-flow routes for every OD pair."""
    cache = {} if cache is None else cache
    if "graph" not in cache:
        cache["graph"] = _routing_graph(edges)
    graph = cache["graph"]

    def paths(source, destination, limit):
        key = ("paths", int(source), int(destination), int(limit))
        if key not in cache:
            cache[key] = _shortest_paths(graph, source, destination, limit)
        return cache[key]
    od_pairs = sorted({(int(record.origin), int(record.destination)) for record in demand_classes})
    chargers = tuple(sorted(int(value) for value in chargers))
    output = {}
    # K leg paths give at least K combinations per charger in ordinary cases.
    leg_limit = max(int(k), 1)
    for od in od_pairs:
        non_key = ("records", "F1", od, int(k))
        if non_key not in cache:
            cache[non_key] = [
                _path_record(graph, path, od=od, vehicle_type="F1", rank=index)
                for index, path in enumerate(paths(*od, int(k)))
            ]
        non_charging = [dict(route) for route in cache[non_key]]
        if not non_charging:
            raise ValueError(f"No feasible non-charging route for OD {od}")
        by_charger = {}
        for charger in chargers:
            record_key = ("records", "F2", od, charger, leg_limit)
            if record_key not in cache:
                first = paths(od[0], charger, leg_limit)
                second = paths(charger, od[1], leg_limit)
                candidates, seen = [], set()
                for left in first:
                    for right in second:
                        path = tuple(int(value) for value in left + right[1:])
                        if len(path) != len(set(path)) or path in seen:
                            continue
                        seen.add(path)
                        candidates.append(_path_record(
                            graph, path, od=od, vehicle_type="F2",
                            charger=charger, rank=len(candidates),
                        ))
                cache[record_key] = candidates
            by_charger[charger] = [dict(route) for route in cache[record_key]]
        charging = balanced_charger_routes(
            by_charger, int(k),
            rank_key=lambda route: (route["free_flow_time"], route["route_id"]),
        )
        output[od] = {
            "no charging type": non_charging[:int(k)],
            "charging type": charging,
        }
    return output


def initialize_counts(routes, total, method, seed_manager=None, seed_key=()):
    """Initialize integer route counts without changing aggregate demand."""
    total = int(total)
    counts = [0] * len(routes)
    if not routes or total <= 0:
        return counts
    if method == "free_flow_shortest":
        best = min(range(len(routes)), key=lambda i: (
            float(routes[i].get("free_flow_time", float("inf"))),
            str(routes[i].get("route_id", i)),
        ))
        counts[best] = total
        return counts
    if method == "seeded_random":
        if seed_manager is None:
            raise ValueError("seeded_random initialization requires a SeedManager")
        draws = seed_manager.numpy("queue-initialization", *seed_key).integers(
            0, len(routes), size=total
        )
        return [int((draws == index).sum()) for index in range(len(routes))]
    if method == "uniform":
        base, remainder = divmod(total, len(routes))
        return [base + int(index < remainder) for index in range(len(routes))]
    raise ValueError(f"Unsupported non-CG initialization: {method}")
