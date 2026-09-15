#!/usr/bin/env python3
"""Run or summarize a generic manifest of independent experiment configs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import traceback
import copy
import shutil
import zipfile
import tempfile
from pathlib import Path

import numpy as np

from src.config import Config
from src.run_state import atomic_write_json, config_digest, safe_name


def _set_override(config, dotted_key, value):
    target = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = copy.deepcopy(value)


def load_manifest(path, expanded_dir=None):
    manifest_path = Path(path).resolve()
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    jobs = []
    if manifest.get("experiments") is not None:
        entries = manifest.get("experiments")
        if not isinstance(entries, list) or not entries:
            raise ValueError("suite experiments must be a non-empty list")
        seen = set()
        for entry in entries:
            job_id = str(entry.get("id", "")).strip()
            if not job_id or job_id in seen:
                raise ValueError(f"suite experiment id is missing or duplicated: {job_id!r}")
            source = entry.get("base", manifest.get("base_config"))
            if not source:
                raise ValueError(f"experiment {job_id} has no base configuration")
            source_path = (manifest_path.parent / source).resolve()
            if not source_path.is_file():
                raise FileNotFoundError(f"suite config does not exist: {source_path}")
            raw = json.loads(source_path.read_text())
            for key, value in sorted(entry.get("overrides", {}).items()):
                _set_override(raw, key, value)
            raw["name"] = job_id
            Config.from_dict(raw)
            dependencies = [entry.get("reuse_network_from"), entry.get("reuse_bpr_from")]
            for dependency in (value for value in dependencies if value):
                if dependency not in seen:
                    raise ValueError(
                        f"experiment {job_id} dependency {dependency!r} must precede it"
                    )
            jobs.append({"id": job_id, "raw": raw, **{
                key: entry.get(key) for key in ("reuse_network_from", "reuse_bpr_from")
            }})
            seen.add(job_id)
    else:
        entries = manifest.get("configs")
        if not isinstance(entries, list) or not entries:
            raise ValueError("suite manifest must contain configs or experiments")
        for entry in entries:
            value = entry["path"] if isinstance(entry, dict) else entry
            config_path = (manifest_path.parent / value).resolve()
            if not config_path.is_file():
                raise FileNotFoundError(f"suite config does not exist: {config_path}")
            raw = json.loads(config_path.read_text())
            Config.from_dict(raw)
            jobs.append({"id": str(raw.get("name", config_path.stem)), "raw": raw,
                         "source_path": config_path,
                         "reuse_network_from": None, "reuse_bpr_from": None})
    if expanded_dir is not None:
        expanded_dir = Path(expanded_dir)
        expanded_dir.mkdir(parents=True, exist_ok=True)
        for index, job in enumerate(jobs):
            path_out = expanded_dir / f"{index:02d}_{safe_name(job['id'])}.json"
            atomic_write_json(path_out, job["raw"])
            job["path"] = path_out
    return manifest, jobs


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


def _complete_run(path):
    status_path = Path(path) / "status.json"
    if not status_path.is_file():
        return False
    try:
        return str(json.loads(status_path.read_text()).get("status", "")).startswith("complete")
    except (OSError, ValueError):
        return False


def run_job(job, jobs_by_id, results_root, resume):
    target = expected_run_dir(job["path"], results_root)
    if resume and _complete_run(target):
        print(f"Skipping completed experiment {job['id']}: {target}")
        return str(target)
    env_updates = {}
    for field, env_name, subdir in (
        ("reuse_network_from", "EVOPT_NETWORK_ARTIFACT", "network"),
        ("reuse_bpr_from", "EVOPT_BPR_ARTIFACT", "bpr"),
    ):
        dependency = job.get(field)
        if dependency:
            dependency_job = jobs_by_id[dependency]
            dependency_dir = expected_run_dir(dependency_job["path"], results_root)
            if not _complete_run(dependency_dir):
                raise RuntimeError(
                    f"{job['id']} requires completed experiment {dependency}"
                )
            env_updates[env_name] = str(dependency_dir / subdir)
    previous = {key: os.environ.get(key) for key in env_updates}
    try:
        os.environ.update(env_updates)
        return run_one(job["path"], results_root, resume)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def select_jobs(jobs, *, index=None, start_index=None):
    """Select one job or a suffix; indices are zero-based."""
    if index is not None and start_index is not None:
        raise ValueError("--index and --start-index cannot be used together")
    if index is not None:
        if index < 0 or index >= len(jobs):
            raise IndexError(f"experiment index {index} outside 0..{len(jobs) - 1}")
        return jobs[index:index + 1]
    if start_index is not None:
        if start_index < 0 or start_index > len(jobs):
            raise IndexError(
                f"start index {start_index} outside 0..{len(jobs)}"
            )
        return jobs[start_index:]
    return jobs


def _record_suite_failure(job, results_root, exc):
    """Persist failures raised before the pipeline itself can write status."""
    target = expected_run_dir(job["path"], results_root)
    target.mkdir(parents=True, exist_ok=True)
    status_path = target / "status.json"
    prior = {}
    if status_path.is_file():
        try:
            prior = json.loads(status_path.read_text())
        except (OSError, ValueError):
            pass
    prior.update({
        "status": "failed",
        "eligible": False,
        "failure_type": type(exc).__name__,
        "failure_reason": str(exc),
        "suite_traceback": traceback.format_exc(),
        "config": str(job["path"]),
        "config_name": job["raw"].get("name", job["id"]),
    })
    atomic_write_json(status_path, prior)


def run_jobs(jobs, jobs_by_id, results_root, resume, *, continue_on_failure=False):
    """Run selected jobs in order and return failures after the requested policy."""
    failures = []
    for job in jobs:
        try:
            run_job(job, jobs_by_id, results_root, resume)
        except Exception as exc:
            _record_suite_failure(job, results_root, exc)
            failures.append((job["id"], exc))
            print(f"Experiment {job['id']} failed: {exc}")
            if not continue_on_failure:
                break
            print("Continuing with the next suite experiment.")
    return failures


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


def _write_csv(path, rows, default_columns):
    columns = list(default_columns)
    for row in rows:
        for column in row:
            if column not in columns:
                columns.append(column)
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


SENSITIVITY_PHYSICAL_EXPERIMENTS = (
    "sensitivity_base", "budget_6_3", "budget_6_4", "budget_7_2",
    "budget_7_3", "budget_7_4", "demand_090", "demand_270",
    "f2_share_1_3", "f2_share_1_2", "od_count_2",
)

SENSITIVITY_FACTORS = (
    ("Sites", "$(6/2)\\ldots(7/4)$",
     ("sensitivity_base", "budget_6_3", "budget_6_4", "budget_7_2",
      "budget_7_3", "budget_7_4")),
    ("Demand", "$90/180/270$",
     ("demand_090", "sensitivity_base", "demand_270")),
    ("Charge share", "$1/3,1/2,2/3$",
     ("f2_share_1_3", "f2_share_1_2", "sensitivity_base")),
    ("ODs", "$1/2$", ("sensitivity_base", "od_count_2")),
    ("Routes", "$K=8/16/32$",
     ("routes_k08", "sensitivity_base", "routes_k32")),
    ("Route src.", "CG/indep.",
     ("sensitivity_base", "route_source_independent")),
    ("Init.", "CG/unif./rand.",
     ("sensitivity_base", "init_uniform", "init_random")),
    ("BR reps.", "$1/10/50$",
     ("sensitivity_base", "ne_reps_10", "ne_reps_50")),
    ("Q seed", "$42/43/44$",
     ("sensitivity_base", "queue_seed_43", "queue_seed_44")),
)

REPORTING_STRATEGIES = (
    ("greedy", "Greedy"),
    ("single_swap", "Greedy + swap"),
    ("minimum_detour", "Min. detour"),
    ("weighted_betweenness", "Wt. betw."),
    ("uniform_random_expectation", "Random exp."),
)


def _latest_completed_records(records, experiment_ids=None):
    """Select the newest completed run for each experiment name."""
    allowed = set(experiment_ids) if experiment_ids is not None else None
    selected = {}
    for record in records:
        name = record.get("name")
        if not name or (allowed is not None and name not in allowed):
            continue
        if not str(record.get("status", "")).startswith("complete"):
            continue
        previous = selected.get(name)
        key = (str(record.get("timestamp_utc") or ""), str(record.get("run") or ""))
        old_key = (
            str(previous.get("timestamp_utc") or ""), str(previous.get("run") or "")
        ) if previous else None
        if previous is None or key > old_key:
            selected[name] = record
    return selected


def _is_exact_optimum(strategy):
    gap = strategy.get("regret_pct") if strategy else None
    try:
        return np.isfinite(float(gap)) and abs(float(gap)) <= 1e-9
    except (TypeError, ValueError):
        return False


def _sensitivity_paper_rows(selected):
    """Build machine-readable rows for the horizontal paper table."""
    if "sensitivity_base" not in selected:
        return []
    rows = []
    physical = [selected[name] for name in SENSITIVITY_PHYSICAL_EXPERIMENTS
                if name in selected]
    for strategy, label in REPORTING_STRATEGIES:
        rows.append({
            "panel": "A_placement", "factor": label, "runs": len(physical),
            "cg_exact": sum(_is_exact_optimum(value["cg_strategies"].get(strategy))
                            for value in physical),
            "queue_exact": sum(
                _is_exact_optimum(value["queue_strategies"].get(strategy))
                for value in physical
            ),
        })

    for factor, levels, names in SENSITIVITY_FACTORS:
        values = [selected[name] for name in names if name in selected]
        correlations = [float(value["spearman"]) for value in values
                        if value.get("spearman") is not None
                        and np.isfinite(float(value["spearman"]))]
        agreements = [bool(value["top_1_agreement"]) for value in values
                      if value.get("top_1_agreement") is not None]
        def mean_minutes(*fields):
            timings = [sum(float(value.get(field) or 0) for field in fields)
                       for value in values
                       if all(value.get(field) is not None for field in fields)]
            return float(np.mean(timings)) / 60 if timings else None
        rows.append({
            "panel": "B_agreement", "factor": factor, "levels": levels,
            "runs": len(values),
            "spearman_median": float(np.median(correlations)) if correlations else None,
            "spearman_min": min(correlations, default=None),
            "spearman_max": max(correlations, default=None),
            "top_1_agreement_count": sum(agreements),
            "top_1_agreement_rate": sum(agreements) / len(agreements) if agreements else None,
            "mean_bpr_minutes": (
                sum(float(value.get("bpr_seconds") or 0)
                    for value in values if value.get("name") == "sensitivity_base")
                / len(values) / 60 if values else None
            ),
            "mean_cg_minutes": mean_minutes("cg_seconds"),
            "mean_queue_minutes": mean_minutes("queue_ne_seconds"),
            "mean_total_minutes": mean_minutes("total_seconds"),
        })

    values = list(selected.values())
    correlations = [float(value["spearman"]) for value in values
                    if value.get("spearman") is not None
                    and np.isfinite(float(value["spearman"]))]
    agreements = [bool(value["top_1_agreement"]) for value in values
                  if value.get("top_1_agreement") is not None]
    def overall_mean_minutes(*fields):
        timings = [sum(float(value.get(field) or 0) for field in fields)
                   for value in values
                   if all(value.get(field) is not None for field in fields)]
        return float(np.mean(timings)) / 60 if timings else None
    rows.append({
        "panel": "B_agreement", "factor": "Overall", "levels": "All selected jobs",
        "runs": len(values),
        "spearman_median": float(np.median(correlations)) if correlations else None,
        "spearman_min": min(correlations, default=None),
        "spearman_max": max(correlations, default=None),
        "top_1_agreement_count": sum(agreements),
        "top_1_agreement_rate": sum(agreements) / len(agreements) if agreements else None,
        "mean_bpr_minutes": (
            sum(float(value.get("bpr_seconds") or 0)
                for value in values if value.get("name") == "sensitivity_base")
            / len(values) / 60 if values else None
        ),
        "mean_cg_minutes": overall_mean_minutes("cg_seconds"),
        "mean_queue_minutes": overall_mean_minutes("queue_ne_seconds"),
        "mean_total_minutes": overall_mean_minutes("total_seconds"),
    })
    return rows


def _render_sensitivity_table_tex(rows, selected):
    """Render placement and sensitivity results side by side."""
    placement_rows = [row for row in rows if row["panel"] == "A_placement"]
    sensitivity_rows = [row for row in rows if row["panel"] == "B_agreement"]
    node_counts = sorted({int(value["nodes"]) for value in selected.values()
                          if value.get("nodes") is not None})
    edge_counts = sorted({int(value["edges"]) for value in selected.values()
                          if value.get("edges") is not None})
    network = (f"{node_counts[0]}-node, {edge_counts[0]}-link "
               if len(node_counts) == len(edge_counts) == 1 else "")
    cycles = sum(int(value.get("cycles") or 0) for value in selected.values())
    searches = sum(int(value.get(field) or 0) for value in selected.values()
                   for field in ("cycles", "converged", "nonconverged", "failed"))
    physical_runs = placement_rows[0]["runs"] if placement_rows else 0

    lines = [
        "% Generated by run_suite.py; requires \\usepackage{booktabs}.",
        "\\begin{table*}[t]",
        "  \\centering",
        f"  \\caption{{Placement quality, cross-model agreement, and mean runtime on the {network}DC sensitivity network.}}",
        "  \\label{tab:rebuttal_sensitivity}",
        "  \\normalsize",
        "  \\setlength{\\tabcolsep}{1pt}",
        "  \\renewcommand{\\arraystretch}{1.10}",
        "  \\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}lcc@{\\hspace{0.2em}}lccr@{}}",
        "    \\toprule",
        "    \\multicolumn{3}{c}{Placement quality} & \\multicolumn{4}{c}{Sensitivity and runtime} \\\\",
        "    \\cmidrule(lr){1-3} \\cmidrule(lr){4-7}",
        "    Method & CG & Q & Setting & $\\rho$ [range] & Top-1 & BPR/CG/Q/Total \\\\",
        "    \\midrule",
    ]
    for index in range(max(len(placement_rows), len(sensitivity_rows))):
        placement = placement_rows[index] if index < len(placement_rows) else None
        sensitivity = sensitivity_rows[index] if index < len(sensitivity_rows) else None
        if sensitivity and sensitivity["factor"] == "Overall":
            lines.append("    \\cmidrule(lr){4-7}")
        method = placement["factor"] if placement else ""
        cg_exact = f"{placement['cg_exact']}/{placement['runs']}" if placement else ""
        q_exact = f"{placement['queue_exact']}/{placement['runs']}" if placement else ""
        setting = ""
        if sensitivity:
            setting = sensitivity["factor"]
            if sensitivity["factor"] != "Overall":
                setting += f" {sensitivity['levels']}"
        median = sensitivity.get("spearman_median") if sensitivity else None
        correlation = "--" if median is None else (
            f"${median:.2f}\\,[{sensitivity['spearman_min']:.2f},"
            f"{sensitivity['spearman_max']:.2f}]$"
        )
        rate = sensitivity.get("top_1_agreement_rate") if sensitivity else None
        top_one = "--" if rate is None else (
            f"{sensitivity['top_1_agreement_count']}/{sensitivity['runs']} "
            f"({100 * rate:.0f}\\%)"
        )
        timing = ""
        if sensitivity:
            timing = "/".join(
                f"{sensitivity[field]:.2f}" for field in (
                    "mean_bpr_minutes", "mean_cg_minutes",
                    "mean_queue_minutes", "mean_total_minutes",
                )
            )
        lines.append(
            f"    {method} & {cg_exact} & {q_exact} & {setting} & "
            f"{correlation} & {top_one} & {timing} \\\\"
        )
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular*}",
        "  \\vspace{3pt}",
        "  \\begin{minipage}{0.985\\textwidth}",
        "    \\footnotesize",
        f"    \\textit{{Notes:}} Placement quality uses {physical_runs} physical-scenario variants; sensitivity statistics use {len(selected)} selected jobs. Exact denotes equality with the exhaustive objective within $10^{{-9}}$ percentage points; random refers to its expected objective. CG is the congestion game; Q is the queue cycle-state approximation; Top-1 requires the same best charger set. BPR/CG/Q/Total are mean wall minutes; BPR includes fitting only (shared-artifact loading is excluded), and Q includes queue search only (final comparison is excluded). Total is the complete pipeline and additionally includes final comparison and overhead. "
        f"All {cycles}/{searches} queue searches cycled and are not Nash equilibria. Times are not CPU-hours; BPR was calibrated once and reused by the other jobs.",
        "  \\end{minipage}",
        "\\end{table*}",
        "",
    ])
    return "\n".join(lines)


def export_bundle(results_root, destination, experiment_ids=None):
    """Export compact reviewer-facing tables/figures without raw simulations."""
    root = Path(results_root)
    runs = summarize(root)
    placements, queue_rows, bpr_rows = [], [], []
    summaries, statuses = {}, {}
    run_dirs = [path.parent for path in sorted(root.glob("*/status.json"))]
    with tempfile.TemporaryDirectory(prefix="evopt-bundle-") as temporary:
        stage = Path(temporary)
        for run_dir in run_dirs:
            status_path = run_dir / "status.json"
            statuses[run_dir.name] = json.loads(status_path.read_text())
            summary_path = run_dir / "experiment_summary.json"
            summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
            summaries[run_dir.name] = summary
            cg = summary.get("cg_results") or {}
            queue = summary.get("queue_results") or {}
            for item in cg.get("all_configs", []):
                placements.append({
                    "run": run_dir.name, "model": "cg",
                    "placement": json.dumps(sorted(item.get("chargers", []))),
                    "objective": item.get("objective"),
                })
            for item in queue.get("exhaustive_results", []):
                placements.append({
                    "run": run_dir.name, "model": "queue",
                    "placement": json.dumps(sorted(item.get("positions", []))),
                    "objective": item.get("avg_travel_time"),
                })
            for model, strategies in queue.get("reviewer_baselines", {}).items():
                for method, item in strategies.items():
                    placements.append({
                        "run": run_dir.name, "model": model,
                        "placement": json.dumps(item.get("placement")),
                        "objective": item.get("objective"),
                        "strategy": method, "regret_pct": item.get("regret_pct"),
                    })
            queue_manifest = run_dir / "queue" / "queue_manifest.json"
            if queue_manifest.is_file():
                value = json.loads(queue_manifest.read_text())
                counts = value.get("status_counts", {})
                stats = value.get("iteration_statistics", {}).get("all", {})
                queue_rows.append({
                    "run": run_dir.name, "converged": counts.get("converged", 0),
                    "cycles": counts.get("cycle", 0),
                    "nonconverged": counts.get("nonconverged", 0),
                    "failed": counts.get("failed", 0),
                    "iterations_median": stats.get("median"),
                    "iterations_p95": stats.get("p95"),
                    "iterations_max": stats.get("max"),
                    "cycle_length_median": value.get("cycle_length_statistics", {}).get("median"),
                    "final_gap_min_normalized_median": value.get(
                        "final_gap_statistics", {}
                    ).get("median"),
                    "final_gap_mean_normalized_median": value.get(
                        "final_gap_mean_normalized_statistics", {}
                    ).get("median"),
                    "minimum_gap_min_normalized_median": value.get(
                        "minimum_gap_statistics", {}
                    ).get("median"),
                    "minimum_gap_mean_normalized_median": value.get(
                        "minimum_gap_mean_normalized_statistics", {}
                    ).get("median"),
                    "route_source": value.get("route_source"),
                    "initialization": value.get("initialization"),
                    "simulator_seed": value.get("simulator_seed"),
                    "pearson": queue.get("cg_queue_correlations", {}).get("pearson"),
                    "spearman": queue.get("cg_queue_correlations", {}).get("spearman"),
                    "top_1_agreement": queue.get("cg_queue_correlations", {}).get("top_1_agreement"),
                    "top_3_overlap_count": queue.get("cg_queue_correlations", {}).get("top_3_overlap_count"),
                })
            bpr_manifest = run_dir / "bpr" / "bpr_manifest.json"
            if bpr_manifest.is_file():
                value = json.loads(bpr_manifest.read_text())
                bpr_rows.append({
                    "run": run_dir.name, "network_hash": value.get("network_hash"),
                    "method": value.get("fitter_version"),
                    "fit_status_counts": json.dumps(value.get("fit_status_counts", {}), sort_keys=True),
                    "r2_summary": json.dumps(value.get("r2_summary", {}), sort_keys=True),
                    "below_configured_r2_count": value.get("below_configured_r2_count"),
                    "seed": value.get("random_seed"),
                })
        run_by_name = {row["run"]: row for row in runs}
        bpr_by_name = {row["run"]: row for row in bpr_rows}
        candidate_records = []
        for run_name, summary in summaries.items():
            config = summary.get("config", {})
            status = statuses.get(run_name, {})
            name = config.get("name", status.get("config_name", run_name))
            queue = summary.get("queue_results") or {}
            baselines = queue.get("reviewer_baselines", {})
            qbase = baselines.get("queue", {})
            cgbase = baselines.get("congestion_game", {})
            manifest_row = run_by_name.get(run_name, {})
            qstats = queue.get("ne_statistics", {}).get("status_counts", {})
            correlations = queue.get("cg_queue_correlations", {})
            demand = config.get("scenario_generation", {}).get("demand", {})
            known_time = sum(float(manifest_row.get(field) or 0) for field in (
                "bpr_seconds", "cg_seconds", "queue_ne_seconds",
                "queue_comparison_seconds",
            ))
            total_time = manifest_row.get("total_seconds")
            candidate_records.append({
                "name": name, "run": run_name,
                "status": status.get("status", manifest_row.get("status")),
                "timestamp_utc": status.get("timestamp_utc"),
                "config_digest": status.get("config_digest"),
                "cg_strategies": cgbase, "queue_strategies": qbase,
                "queue_optimum": json.dumps(qbase.get("exhaustive", {}).get("placement")),
                "cycles": qstats.get("cycle", 0),
                "converged": qstats.get("converged", 0),
                "nonconverged": qstats.get("nonconverged", 0),
                "failed": qstats.get("failed", 0),
                "pearson": correlations.get("pearson"),
                "spearman": correlations.get("spearman"),
                "top_1_agreement": correlations.get("top_1_agreement"),
                "nodes": manifest_row.get("nodes"), "edges": manifest_row.get("edges"),
                "total_seconds": total_time,
                "bpr_seconds": manifest_row.get("bpr_seconds"),
                "cg_seconds": manifest_row.get("cg_seconds"),
                "queue_ne_seconds": manifest_row.get("queue_ne_seconds"),
                "queue_comparison_seconds": manifest_row.get("queue_comparison_seconds"),
                "other_seconds": max(0.0, float(total_time) - known_time)
                if total_time is not None else None,
                "bpr_fit_status_counts": bpr_by_name.get(run_name, {}).get("fit_status_counts"),
                "demand": int(demand.get("F1", 0)) + int(demand.get("F2", 0)),
                "configurations": (summary.get("cg_results") or {}).get("num_configs"),
            })

        selected = _latest_completed_records(candidate_records, experiment_ids)
        scaling_rows = []
        for name, value in selected.items():
            if not str(name).startswith("secondary-plus-scale-"):
                continue
            scaling_rows.append({
                "panel": "A_scaling", "factor": name,
                "run": value["run"], "nodes": value.get("nodes"),
                "edges": value.get("edges"), "demand": value.get("demand"),
                "configurations": value.get("configurations"),
                "cg_greedy_gap": value["cg_strategies"].get("greedy", {}).get("regret_pct"),
                "cg_swap_gap": value["cg_strategies"].get("single_swap", {}).get("regret_pct"),
                "queue_greedy_gap": value["queue_strategies"].get("greedy", {}).get("regret_pct"),
                "queue_swap_gap": value["queue_strategies"].get("single_swap", {}).get("regret_pct"),
                "queue_optimum": value.get("queue_optimum"),
                "cycles": value.get("cycles"), "converged": value.get("converged"),
                "pearson": value.get("pearson"), "spearman": value.get("spearman"),
                "total_seconds": value.get("total_seconds"),
                "bpr_seconds": value.get("bpr_seconds"),
                "cg_seconds": value.get("cg_seconds"),
                "queue_ne_seconds": value.get("queue_ne_seconds"),
                "queue_comparison_seconds": value.get("queue_comparison_seconds"),
                "bpr_fit_status_counts": value.get("bpr_fit_status_counts"),
            })
        sensitivity_rows = _sensitivity_paper_rows(selected)
        paper_rows = scaling_rows + sensitivity_rows
        cohort = {
            name: {
                "run": value["run"], "config_digest": value.get("config_digest"),
                "status": value.get("status"), "timestamp_utc": value.get("timestamp_utc"),
            }
            for name, value in sorted(selected.items())
        }

        _write_csv(stage / "runs.csv", runs, ["run", "status", "eligible"])
        _write_csv(stage / "placements.csv", placements, ["run", "model", "placement", "objective"])
        _write_csv(stage / "queue_diagnostics.csv", queue_rows, ["run", "converged", "cycles"])
        _write_csv(stage / "bpr_summary.csv", bpr_rows, ["run", "network_hash", "method"])
        _write_csv(stage / "paper_table.csv", paper_rows, ["panel", "factor", "runs"])
        atomic_write_json(stage / "reporting_cohort.json", cohort)
        if sensitivity_rows:
            (stage / "rebuttal_sensitivity_table.tex").write_text(
                _render_sensitivity_table_tex(sensitivity_rows, selected)
            )

        try:
            import matplotlib.pyplot as plt
            completed = [row for row in runs if row.get("total_seconds") is not None]
            strategy_rows = [row for row in placements if row.get("strategy")]
            if strategy_rows:
                methods = sorted({row["strategy"] for row in strategy_rows})
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
                for axis, model in zip(axes, ("congestion_game", "queue")):
                    values = [
                        [float(row["regret_pct"]) for row in strategy_rows
                         if row["model"] == model and row["strategy"] == method]
                        for method in methods
                    ]
                    axis.boxplot(values, tick_labels=[value.replace('_', '\n') for value in methods],
                                 showfliers=False)
                    axis.axhline(0, color="black", linewidth=.8)
                    axis.set_title("Congestion game" if model == "congestion_game" else "Queue simulation")
                    axis.tick_params(axis="x", labelrotation=25)
                axes[0].set_ylabel("Regret from exhaustive optimum (%)")
                fig.tight_layout()
                fig.savefig(stage / "placement_quality.png", dpi=220)
                plt.close(fig)
            if completed:
                completed.sort(key=lambda row: float(row.get("nodes") or 0))
                labels = [str(row.get("nodes") or row["run"]) for row in completed]
                stages = ("bpr_seconds", "cg_seconds", "queue_ne_seconds", "queue_comparison_seconds")
                bottom = [0.0] * len(completed)
                fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
                ax = axes[2]
                for field in stages:
                    values = [float(row.get(field) or 0) / 3600 for row in completed]
                    ax.bar(labels, values, bottom=bottom, label=field.removesuffix("_seconds"))
                    bottom = [a + b for a, b in zip(bottom, values)]
                ax.set(xlabel="Network nodes", ylabel="Wall time (hours)", title="Pipeline scaling")
                ax.legend(frameon=True, ncol=2)
                for run_dir in run_dirs:
                    summary_path = run_dir / "experiment_summary.json"
                    if not summary_path.is_file():
                        continue
                    value = json.loads(summary_path.read_text())
                    for history in (value.get("ne_convergence") or {}).values():
                        axes[0].plot(range(1, len(history) + 1), history, alpha=.18, linewidth=.8)
                axes[0].axhline(.01, color="black", linestyle="--", linewidth=1, label="α = 1%")
                axes[0].set(xlabel="Better-response iteration", ylabel="Relative gap",
                            title="Queue search trajectories")
                axes[0].legend(frameon=True)
                corr = [row for row in queue_rows if row.get("pearson") is not None]
                positions = np.arange(len(corr)) if corr else np.arange(0)
                if corr:
                    axes[1].scatter(positions, [row["pearson"] for row in corr], label="Pearson")
                    axes[1].scatter(positions, [row["spearman"] for row in corr], label="Spearman")
                axes[1].axhline(0, color="black", linewidth=.8)
                axes[1].set(xlabel="Experiment", ylabel="Correlation", title="CG–queue ranking agreement")
                axes[1].legend(frameon=True)
                fig.tight_layout()
                fig.savefig(stage / "robustness_scalability.png", dpi=220)
                plt.close(fig)
        except Exception as exc:
            (stage / "figure_generation_warning.txt").write_text(str(exc) + "\n")

        for run_dir in run_dirs:
            for filename in ("resolved_config.json", "run_manifest.json", "status.json"):
                source = run_dir / filename
                if source.is_file():
                    target = stage / "provenance" / run_dir.name
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, target / filename)
        checksum = {}
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                import hashlib
                checksum[str(path.relative_to(stage))] = hashlib.sha256(path.read_bytes()).hexdigest()
        atomic_write_json(stage / "checksums.json", checksum)
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(stage.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(stage))
    if destination.stat().st_size > 25 * 1024 * 1024:
        raise ValueError(f"Export bundle exceeds 25 MiB: {destination}")
    print(f"Exported compact result bundle: {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--index", type=int, help="zero-based config index; defaults to SLURM_ARRAY_TASK_ID or all")
    parser.add_argument("--start-index", type=int,
                        help="zero-based config index; run this job and all later jobs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true",
                        help="record a failed job and continue with later suite jobs")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--export-bundle")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--print-count", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--first-pending", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    expanded_dir = Path(args.results_root) / ".suite_configs" / safe_name(
        Path(args.manifest).stem
    )
    _manifest, jobs = load_manifest(args.manifest, expanded_dir=expanded_dir)
    if args.print_count:
        print(len(jobs))
        return
    if args.first_pending:
        first = next((
            index for index, job in enumerate(jobs)
            if not _complete_run(expected_run_dir(job["path"], args.results_root))
        ), len(jobs))
        print(first)
        return
    if args.validate_only:
        print(f"Validated {len(jobs)} configuration(s)")
        return
    if args.summarize:
        summarize(args.results_root)
        if args.export_bundle:
            export_bundle(
                args.results_root, args.export_bundle,
                experiment_ids=[job["id"] for job in jobs],
            )
        return
    index = args.index
    if index is None and os.environ.get("SLURM_ARRAY_TASK_ID") is not None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    try:
        selected = select_jobs(jobs, index=index, start_index=args.start_index)
    except (IndexError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    jobs_by_id = {job["id"]: job for job in jobs}
    failures = run_jobs(
        selected, jobs_by_id, args.results_root, args.resume,
        continue_on_failure=args.continue_on_failure,
    )
    summarize(args.results_root)
    if args.export_bundle:
        export_bundle(
            args.results_root, args.export_bundle,
            experiment_ids=[job["id"] for job in jobs],
        )
    if failures:
        failed_ids = ", ".join(job_id for job_id, _exc in failures)
        raise SystemExit(f"Suite finished with {len(failures)} failed experiment(s): {failed_ids}")


if __name__ == "__main__":
    main()
