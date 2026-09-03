"""Canonical, portable network artifact serialization."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.contracts import stable_json


ARTIFACT_SCHEMA_VERSION = 1


def _records(df: pd.DataFrame, sort_column: str) -> list[dict[str, Any]]:
    if df is None:
        return []
    ordered = df.sort_values(sort_column, kind="mergesort").reset_index(drop=True)
    return json.loads(stable_json(ordered.to_dict(orient="records")))


def _canonical_csv(df: pd.DataFrame, sort_column: str) -> bytes:
    """Use CSV normalization so hashes survive a write/read round trip."""
    ordered = df.sort_values(sort_column, kind="mergesort").reset_index(drop=True)
    return ordered.to_csv(index=False, lineterminator="\n", na_rep="").encode("utf-8")


def compute_network_hash(nodes: pd.DataFrame, edges: pd.DataFrame, source: Mapping[str, Any] | None = None) -> str:
    return _hash_bytes(
        _canonical_csv(nodes, "node_id"),
        _canonical_csv(edges, "link_id"),
        source,
    )


def _hash_bytes(nodes_bytes: bytes, edges_bytes: bytes, source: Mapping[str, Any] | None = None) -> str:
    hasher = hashlib.sha256()
    hasher.update(stable_json({
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "source": dict(source or {}),
    }).encode("utf-8"))
    hasher.update(b"\nNODES\n")
    hasher.update(nodes_bytes)
    hasher.update(b"\nEDGES\n")
    hasher.update(edges_bytes)
    return hasher.hexdigest()


def write_network_artifact(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    output_dir: str | os.PathLike[str],
    stage_counts: Mapping[str, Any] | None = None,
    stage_maps: Mapping[str, Any] | None = None,
    source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write deterministic CSVs and a manifest, returning the manifest."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    nodes_path = root / "nodes.csv"
    edges_path = root / "edges.csv"
    manifest_path = root / "network_manifest.json"
    stage_maps_path = root / "stage_maps.json"

    nodes_out = nodes.sort_values("node_id", kind="mergesort").reset_index(drop=True)
    edges_out = edges.sort_values("link_id", kind="mergesort").reset_index(drop=True)
    nodes_out.to_csv(nodes_path, index=False)
    edges_out.to_csv(edges_path, index=False)

    source_data = dict(source or {})
    network_hash = _hash_bytes(nodes_path.read_bytes(), edges_path.read_bytes(), source_data)
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "network_hash": network_hash,
        "nodes_file": nodes_path.name,
        "edges_file": edges_path.name,
        "node_count": int(len(nodes_out)),
        "edge_count": int(len(edges_out)),
        "node_id_policy": "deterministic contiguous integer after final SCC",
        "edge_id_policy": "deterministic contiguous integer after final SCC",
        "edge_identity": ["link_id", "start_node_id", "end_node_id", "edge_key"],
        "node_id_mapping": [
            {
                "node_id": int(row.node_id),
                "source_node_id": str(getattr(row, "node_osmid", row.node_id)),
            }
            for row in nodes_out.itertuples()
        ],
        "edge_id_mapping": [
            {
                "link_id": int(row.link_id),
                "source_start": str(getattr(row, "start_osmid", row.start_node_id)),
                "source_end": str(getattr(row, "end_osmid", row.end_node_id)),
                "source_key": str(getattr(row, "edge_key", "")),
            }
            for row in edges_out.itertuples()
        ],
        "source": source_data,
        "stage_counts": dict(stage_counts or {}),
        "stage_map_names": sorted((stage_maps or {}).keys()),
    }
    with open(manifest_path, "w") as handle:
        json.dump(manifest, handle, indent=2, default=str)
    if stage_maps:
        with open(stage_maps_path, "w") as handle:
            json.dump(stage_maps, handle, indent=2, default=str)
        manifest["stage_maps_file"] = stage_maps_path.name
        with open(manifest_path, "w") as handle:
            json.dump(manifest, handle, indent=2, default=str)
    return manifest


def load_network_artifact(path: str | os.PathLike[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = Path(path)
    manifest_path = root / "network_manifest.json" if root.is_dir() else root
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    base = manifest_path.parent
    nodes = pd.read_csv(base / manifest["nodes_file"])
    edges = pd.read_csv(base / manifest["edges_file"])
    actual = _hash_bytes(
        (base / manifest["nodes_file"]).read_bytes(),
        (base / manifest["edges_file"]).read_bytes(),
        manifest.get("source", {}),
    )
    if actual != manifest.get("network_hash"):
        raise ValueError(
            f"Network artifact hash mismatch: expected {manifest.get('network_hash')}, got {actual}"
        )
    required_nodes = {"node_id", "lon", "lat"}
    required_edges = {"link_id", "start_node_id", "end_node_id"}
    if not required_nodes.issubset(nodes.columns):
        raise ValueError(f"Network nodes missing columns: {sorted(required_nodes - set(nodes.columns))}")
    if not required_edges.issubset(edges.columns):
        raise ValueError(f"Network edges missing columns: {sorted(required_edges - set(edges.columns))}")
    if edges["link_id"].duplicated().any():
        raise ValueError("Network artifact contains duplicate link_id values")
    return nodes, edges, manifest
