"""Persistent OSM graph cache — downloads once, loads from disk forever after."""
import os
import pickle
import osmnx as ox


GRAPH_CACHE_DIR = "data/graphs"


def _cache_key(coords, highway_types):
    import hashlib, json
    key = json.dumps([list(coords), sorted(highway_types) if highway_types else "all"], sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def get_graph(coords, highway_types=None, force_refresh=False):
    """Get an OSM graph from disk cache, downloading only if absent.

    Args:
        coords: (west, south, east, north) or (north, south, east, west) bbox.
        highway_types: List of OSM highway types, or None for all drivable.
        force_refresh: If True, re-download even if cached.

    Returns:
        networkx.MultiDiGraph
    """
    os.makedirs(GRAPH_CACHE_DIR, exist_ok=True)
    key = _cache_key(coords, highway_types)
    path = os.path.join(GRAPH_CACHE_DIR, f"{key}.pkl")

    if not force_refresh and os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    print(f"  Downloading OSM graph {key}...")
    if highway_types:
        filt = '["highway"~"' + '|'.join(highway_types) + '"]'
        G = ox.graph_from_bbox(coords, custom_filter=filt)
    else:
        G = ox.graph_from_bbox(coords, network_type="drive")

    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached to {path}")
    return G
