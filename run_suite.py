#!/usr/bin/env python3
"""Run or summarize a generic manifest of independent experiment configs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
from pathlib import Path

from src.config import Config
from src.run_state import atomic_write_json, config_digest, safe_name


def load_manifest(path):
    manifest_path = Path(path).resolve()
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    entries = manifest.get("configs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("suite manifest must contain a non-empty configs list")
    configs = []
    for entry in entries:
        value = entry["path"] if isinstance(entry, dict) else entry
        config_path = (manifest_path.parent / value).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"suite config does not exist: {config_path}")
        configs.append(config_path)
    return manifest, configs


def expected_run_dir(config_path, results_root):
    config = Config.from_json(str(config_path))
    digest = config_digest(config.to_dict())
    return Path(results_root) / f"{safe_name(config.name)}-{digest[:12]}"


def run_one(config_path, results_root, resume):
    from pipeline import run_pipeline
    config = Config.from_json(str(config_path))
    target = expected_run_dir(config_path, results_root)
    try:
        return run_pipeline(str(config_path), results_root=str(results_root), resume=resume)
    except Exception as exc:
        target.mkdir(parents=True, exist_ok=True)
        prior = {}
        status_path = target / "status.json"
        if status_path.exists():
            try:
                prior = json.loads(status_path.read_text())
            except (OSError, ValueError):
                prior = {}
        atomic_write_json(status_path, {
            **prior,
            "status": prior.get("status") if prior.get("status") == "ineligible" else "failed",
            "eligible": False,
            "failure_type": type(exc).__name__,
            "failure_reason": str(exc),
            "traceback": traceback.format_exc(),
            "config": str(config_path),
            "config_name": config.name,
        })
        raise


def summarize(results_root):
    root = Path(results_root)
    rows = []
    for status_path in sorted(root.glob("*/status.json")):
        try:
            status = json.loads(status_path.read_text())
        except (OSError, ValueError):
            continue
        run_dir = status_path.parent
        manifest = {}
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, ValueError):
                manifest = {}
        network = manifest.get("network", {})
        timing = manifest.get("timing", status.get("timing", {}))
        rows.append({
            "run": run_dir.name,
            "status": status.get("status", "unknown"),
            "eligible": status.get("eligible", False),
            "stage": status.get("stage"),
            "nodes": network.get("node_count"),
            "edges": network.get("edge_count"),
            "network_hash": status.get("network_hash", network.get("network_hash")),
            "total_seconds": timing.get("total"),
            "bpr_seconds": timing.get("bpr_fitting"),
            "cg_seconds": timing.get("cg_optimization"),
            "queue_ne_seconds": timing.get("queue_ne"),
            "queue_comparison_seconds": timing.get("queue_comparison"),
            "artifact_bytes": status.get("artifact_bytes"),
            "failure_reason": status.get("failure_reason"),
        })
    root.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0]) if rows else ["run", "status", "eligible"]
    with (root / "suite_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Experiment suite summary", "", f"Runs found: {len(rows)}", ""]
    if rows:
        lines.extend([
            "| Run | Status | Eligible | N | E | Total hours | Artifact GiB |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in rows:
            hours = float(row["total_seconds"]) / 3600 if row.get("total_seconds") is not None else float("nan")
            gib = float(row["artifact_bytes"]) / 2**30 if row.get("artifact_bytes") is not None else float("nan")
            lines.append(
                f"| {row['run']} | {row['status']} | {row['eligible']} | "
                f"{row.get('nodes')} | {row.get('edges')} | {hours:.3f} | {gib:.3f} |"
            )
    (root / "suite_summary.md").write_text("\n".join(lines) + "\n")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--index", type=int, help="zero-based config index; defaults to SLURM_ARRAY_TASK_ID or all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    _manifest, configs = load_manifest(args.manifest)
    for path in configs:
        Config.from_json(str(path))
    if args.validate_only:
        print(f"Validated {len(configs)} configuration(s)")
        return
    if args.summarize:
        summarize(args.results_root)
        return
    index = args.index
    if index is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    selected = configs if index is None else [configs[index]]
    failures = []
    for path in selected:
        try:
            run_one(path, args.results_root, args.resume)
        except Exception as exc:
            failures.append((str(path), str(exc)))
    summarize(args.results_root)
    if failures:
        raise SystemExit("; ".join(f"{path}: {reason}" for path, reason in failures))


if __name__ == "__main__":
    main()
