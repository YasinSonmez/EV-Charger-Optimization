"""Persistent OSM graph cache — downloads once, loads from disk forever after."""
import os
import pickle
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import osmnx as ox


GRAPH_CACHE_DIR = "data/graphs"


def graph_cache_dir():
    """Return the mountable cache root used by local and Slurm runs."""
    return os.environ.get("EVOPT_GRAPH_CACHE_DIR", GRAPH_CACHE_DIR)


def _cache_key(coords, highway_types, *, network_type, simplify, retain_all,
               truncate_by_edge, custom_filter):
    import hashlib, json
    key = json.dumps({
        "coordinates": list(coords),
        "highway_types": sorted(highway_types) if highway_types else None,
        "network_type": network_type,
        "simplify": bool(simplify),
        "retain_all": bool(retain_all),
        "truncate_by_edge": bool(truncate_by_edge),
        "custom_filter": custom_filter,
    }, sort_keys=True)
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _drivable_highway_filter(highway_types):
    road_pattern = "|".join(sorted(set(highway_types)))
    return (
        f'["highway"~"{road_pattern}"]'
        '["area"!~"yes"]'
        '["access"!~"private"]'
        '["motor_vehicle"!~"no"]'
        '["motorcar"!~"no"]'
    )


def get_graph(coords, highway_types=None, force_refresh=False, *,
              network_type="drive", simplify=True, retain_all=False,
              truncate_by_edge=False, custom_filter=None, require_cached=False,
              return_metadata=False):
    """Get an OSM graph from disk cache, downloading only if absent.

    Args:
        coords: (west, south, east, north) or (north, south, east, west) bbox.
        highway_types: List of OSM highway types, or None for all drivable.
        force_refresh: If True, re-download even if cached.
        network_type, simplify, retain_all, truncate_by_edge, custom_filter:
            Passed through to OSMnx and included in the cache key. Pruning
            studies use ``simplify=False`` so filtering happens before any
            topological nodes are removed.

    Returns:
        networkx.MultiDiGraph
    """
    cache_dir = graph_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    # OSMnx maintains its own HTTP response cache in addition to our processed
    # graph cache. Point it beneath the same configurable, mounted cache root so
    # container runs never attempt to write into a read-only source checkout.
    osmnx_cache_dir = Path(cache_dir) / "osmnx"
    osmnx_cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = osmnx_cache_dir
    if custom_filter is None and highway_types:
        custom_filter = _drivable_highway_filter(highway_types)
    key = _cache_key(
        coords,
        highway_types,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
        custom_filter=custom_filter,
    )
    path = os.path.join(cache_dir, f"{key}.pkl")
    metadata_path = os.path.join(cache_dir, f"{key}.json")

    if not force_refresh and os.path.exists(path):
        with open(path, "rb") as f:
            graph = pickle.load(f)
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path) as handle:
                metadata = json.load(handle)
        metadata.update({
            "cache_key": key,
            "cache_path": os.path.abspath(path),
            "cache_hit": True,
        })
        return (graph, metadata) if return_metadata else graph

    if require_cached:
        raise FileNotFoundError(
            f"OSM graph cache entry {key} is required but absent at {path}"
        )

    print(f"  Downloading OSM graph {key}...")
    G = ox.graph_from_bbox(
        coords,
        network_type=network_type,
        simplify=simplify,
        retain_all=retain_all,
        truncate_by_edge=truncate_by_edge,
        custom_filter=custom_filter,
    )

    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Cached to {path}")
    with open(path, "rb") as handle:
        checksum = hashlib.sha256(handle.read()).hexdigest()
    metadata = {
        "cache_key": key,
        "cache_path": os.path.abspath(path),
        "cache_sha256": checksum,
        "cache_hit": False,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "coordinates": list(coords),
        "highway_types": sorted(highway_types) if highway_types else None,
        "network_type": network_type,
        "simplify": bool(simplify),
        "retain_all": bool(retain_all),
        "truncate_by_edge": bool(truncate_by_edge),
        "custom_filter": custom_filter,
        "osmnx_version": getattr(ox, "__version__", "unknown"),
        "overpass_endpoint": getattr(getattr(ox, "settings", None), "overpass_url", None),
    }
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return (G, metadata) if return_metadata else G
