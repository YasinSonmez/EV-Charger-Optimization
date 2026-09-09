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


def _feasible_charging_candidates(graph, cache, od, charger, total_k, per_charger_need):
    """Enumerate simple O-charger-D paths, deepening the leg search as needed.

    The `total_k` shortest paths per leg can all pass through the opposite OD
    endpoint, which would wrongly report zero feasible charging routes for
    sparse candidates. The per-leg limit therefore escalates until the
    charger can fill its balanced quota or both leg enumerations are
    exhausted, with a bounded cap for cost control.
    """
    limit = max(int(total_k), 1)
    cap = 8 * max(int(total_k), 1)
    seen = set()
    candidates = []
    while True:
        first_key = ("paths", od[0], charger, limit)
        second_key = ("paths", charger, od[1], limit)
        if first_key not in cache:
            cache[first_key] = _shortest_paths(graph, od[0], charger, limit)
        if second_key not in cache:
            cache[second_key] = _shortest_paths(graph, charger, od[1], limit)
        first, second = cache[first_key], cache[second_key]
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
        if len(candidates) >= per_charger_need or limit >= cap:
            break
        if len(first) < limit and len(second) < limit:
            break
        grown = min(limit * 4, cap)
        if grown <= limit:
            break
        limit = grown
    return candidates


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
        per_charger_need = -(-int(k) // max(len(chargers), 1))
        by_charger = {}
        for charger in chargers:
            record_key = ("records", "F2", od, charger, per_charger_need)
            if record_key not in cache:
                cache[record_key] = _feasible_charging_candidates(
                    graph, cache, od, charger, int(k), per_charger_need,
                )
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
