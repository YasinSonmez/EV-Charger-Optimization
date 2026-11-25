#!/usr/bin/env python3
"""Generate the two reviewer-response LaTeX tables directly from a results folder.

Replaces the combined single-table generator in ``run_suite.py`` (kept intact for
recovery) with three separate, publication-ready tables:

* Table 1 -- placement quality versus exhaustive evaluation (exact counts CG/Q,
  pooled median/max regret in the notes) plus per-stage pipeline wall time and
  BPR calibration quality for each distinct network in the provided roots.
* Table 2 -- solver-side robustness on the fixed baseline problem: one row per
  factor whose levels are pooled (route count, initialization, route source,
  better-response sampling, simulator seed), with met/cycled/searches, median
  final gap, cross-model rho, and the percent change of the evaluated queue
  optimum relative to the baseline run -- all comparable within a row because
  every level solves the same problem.
* Table 3 -- per-problem results: one row per physical scenario (charger
  budget or candidate set, demand level, required-charging share, OD
  structure), each with its own optimal congestion-game and queue objectives
  (normalized to the baseline problem under the same model), greedy regret,
  cross-model agreement, alpha status, and the CG/Q stage wall time in
  minutes. Rows are self-contained and not comparable across rows.

The renderer is driven entirely by what is present under ``--results-root``
(the first root is the primary cohort for placement/robustness statistics;
``--runtime-root`` entries contribute additional runtime rows only), so a new
experiment suite is reflected automatically whenever the same experiment names
appear.  Rows for experiments absent from a root are skipped; failed runs are
excluded from all statistics and listed dynamically in the table notes.

Usage:
    python paper_tables.py --results-root results/suite_new6_500 \
        --runtime-root results/rebuttal_sensitivity_results \
        [--output reviewer_tables.tex] [--stdout]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import statistics
import subprocess
from pathlib import Path

from run_suite import REPORTING_STRATEGIES, _latest_completed_records


PHYSICAL_EXPERIMENTS = (
    "sensitivity_base",
    "budget_6_3", "budget_6_4", "budget_7_2", "budget_7_3", "budget_7_4",
    "demand_045", "demand_120", "demand_180",
    "f2_share_1_3", "f2_share_1_2",
    "od_count_2",
)

# (factor label, ((experiment, level label), ...)).  Each Table 2 row pools the
# listed experiments that are present in the cohort, including the baseline
# scenario (demand 90, F2 share 2/3, one OD pair, six candidates, C=2, K=16,
# CG routes/initialization, one better-response sample, seed 101).  Only
# solver-side factors belong here: every level solves the *same* problem, so
# objectives are comparable across the row.  Rows appear only when at least
# two of their levels are present.
TABLE2_FACTOR_GROUPS = (
    ("Route count $K$", (
        ("routes_k08", "8"), ("sensitivity_base", "16"), ("routes_k32", "32"))),
    ("Route source", (
        ("sensitivity_base", "CG"), ("route_source_independent", "indep."))),
    ("Init.", (
        ("sensitivity_base", "CG"), ("init_uniform", "unif."),
        ("init_random", "rand."))),
    ("BR samples", (
        ("sensitivity_base", "1"), ("ne_reps_10", "10"), ("ne_reps_50", "50"))),
    ("Queue seed", (
        ("sensitivity_base", "42"), ("queue_seed_43", "43"),
        ("queue_seed_44", "44"))),
)

# (problem label, experiment).  One row per physical problem for Table 3; each
# row changes a problem-side factor (charger budget or candidate set, demand
# level, required-charging share, OD structure) from the baseline.  Rows are
# self-contained: their objectives are NOT comparable across rows.
TABLE3_PROBLEMS = (
    ("Base: 6 cand., $C{=}2$", "sensitivity_base"),
    ("6 cand., $C{=}3$", "budget_6_3"),
    ("6 cand., $C{=}4$", "budget_6_4"),
    ("7 cand., $C{=}2$", "budget_7_2"),
    ("7 cand., $C{=}3$", "budget_7_3"),
    ("7 cand., $C{=}4$", "budget_7_4"),
    ("45 veh.", "demand_045"),
    ("120 veh.", "demand_120"),
    ("180 veh.", "demand_180"),
    ("F2 $1/2$", "f2_share_1_2"),
    ("F2 $1/3$", "f2_share_1_3"),
    ("Two OD pairs", "od_count_2"),
)

# Solver re-evaluation of the same placement differs by up to about 0.002% in
# this cohort.  Treat objective differences within that numerical tolerance as
# equal; this is deliberately much tighter than any reported placement gap.
EXACT_RELATIVE_TOLERANCE = 2e-5
EXACT_TOLERANCE_PERCENT = 100.0 * EXACT_RELATIVE_TOLERANCE


def _read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def _fmt(value, digits=2):
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _short(value, digits=2):
    """Compact decimal: two digits and a stripped leading zero (0.85 -> .85)."""
    if value is None:
        return "--"
    text = f"{float(value):.{digits}f}"
    if text.startswith("0."):
        return text[1:]
    if text.startswith("-0."):
        return "-" + text[2:]
    return text


def _fmt_signed(value, digits=2):
    if value is None:
        return "--"
    return f"{float(value):+.{digits}f}"


def _range_cell(median, low, high, digits=2):
    if median is None:
        return "--"
    return f"${_fmt(median, digits)}\\, [{_fmt(low, digits)},{_fmt(high, digits)}]$"


def _signed_range_cell(median, low, high, digits=2):
    if median is None:
        return "--"
    return (f"${_fmt_signed(median, digits)}\\, [{_fmt_signed(low, digits)},"
            f"{_fmt_signed(high, digits)}]$")


def _run_directories(root):
    """Run directories sit at the root, under a ``runs/`` level, or the root
    itself may be a single run directory."""
    paths = sorted(root.glob("*/status.json")) + sorted(
        root.glob("runs/*/status.json"))
    if not paths and (root / "status.json").is_file():
        paths = [root / "status.json"]
    return [path.parent for path in paths]


def _provenance_records(root):
    """Minimal timing-only records from archived provenance manifests.

    Provenance directories keep ``run_manifest.json`` (timing, network) and
    ``resolved_config.json`` (demand) but no experiment summaries, so these
    records can only feed runtime rows -- never placement or robustness
    statistics.
    """
    records = []
    for manifest_path in sorted(root.glob("provenance/*/run_manifest.json")):
        run_dir = manifest_path.parent
        manifest = _read_json(manifest_path)
        config = _read_json(run_dir / "resolved_config.json")
        timing = manifest.get("timing") or {}
        network = manifest.get("network") or {}
        demand = (config.get("scenario_generation") or {}).get("demand") or {}
        vehicles = sum(int(demand.get(key) or 0) for key in ("F1", "F2", "F3"))
        records.append({
            "name": run_dir.name,
            "run": run_dir.name,
            "status": "complete",
            "network_hash": network.get("network_hash"),
            "nodes": network.get("node_count"),
            "edges": network.get("edge_count"),
            "vehicles": vehicles or None,
            "total_seconds": timing.get("total"),
            "bpr_seconds": timing.get("bpr_fitting"),
            "cg_seconds": timing.get("cg_optimization"),
            "queue_ne_seconds": timing.get("queue_ne"),
            "queue_comparison_seconds": timing.get("queue_comparison"),
            "cg_strategies": {},
            "queue_strategies": {},
            "pearson": None,
            "spearman": None,
            "top_1_agreement": None,
            "converged": 0,
            "cycles": 0,
            "nonconverged": 0,
            "failed": 0,
            "iterations_median": None,
            "final_gap_median": None,
            "alpha": None,
            "bpr_r2_median": None,
            "bpr_r2_min": None,
        })
    return records


def _load_root_records(results_root, allow_provenance=False):
    """Collect raw records (not deduplicated) for every run in a root."""
    root = Path(results_root)
    records = []
    all_statuses = []
    for run_dir in _run_directories(root):
        status = _read_json(run_dir / "status.json")
        all_statuses.append({"name": run_dir.name, "status": status.get("status")})
        manifest = _read_json(run_dir / "run_manifest.json")
        summary = _read_json(run_dir / "experiment_summary.json")
        config = summary.get("config") or {}
        queue = summary.get("queue_results") or {}
        cg_results = summary.get("cg_results") or {}
        queue_manifest = _read_json(run_dir / "queue" / "queue_manifest.json")
        bpr_manifest = _read_json(run_dir / "bpr" / "bpr_manifest.json")
        timing = manifest.get("timing") or status.get("timing") or {}
        network = manifest.get("network") or {}
        demand = (config.get("scenario_generation") or {}).get("demand") or {}
        counts = queue_manifest.get("status_counts") or {}
        correlations = queue.get("cg_queue_correlations") or {}
        r2 = bpr_manifest.get("r2_summary") or {}
        vehicles = sum(int(demand.get(key) or 0) for key in ("F1", "F2", "F3"))
        records.append({
            "name": config.get("name", run_dir.name),
            "run": run_dir.name,
            "status": status.get("status", "unknown"),
            "network_hash": status.get("network_hash")
            or network.get("network_hash"),
            "nodes": network.get("node_count") or status.get("nodes"),
            "edges": network.get("edge_count") or status.get("edges"),
            "vehicles": vehicles or None,
            "total_seconds": timing.get("total"),
            "bpr_seconds": timing.get("bpr_fitting"),
            "cg_seconds": timing.get("cg_optimization"),
            "queue_ne_seconds": timing.get("queue_ne"),
            "queue_comparison_seconds": timing.get("queue_comparison"),
            "cg_strategies": queue.get("reviewer_baselines", {}).get("congestion_game") or {},
            "queue_strategies": queue.get("reviewer_baselines", {}).get("queue") or {},
            "cg_objectives": [float(item["objective"])
                              for item in (cg_results.get("all_configs") or [])
                              if item.get("objective") is not None],
            "pearson": correlations.get("pearson"),
            "spearman": correlations.get("spearman"),
            "top_1_agreement": correlations.get("top_1_agreement"),
            "converged": int(counts.get("converged") or 0),
            "cycles": int(counts.get("cycle") or 0),
            "nonconverged": int(counts.get("nonconverged") or 0),
            "failed": int(counts.get("failed") or 0),
            "iterations_median": (queue_manifest.get("iteration_statistics") or {})
            .get("all", {}).get("median"),
            "final_gap_median": (queue_manifest.get("final_gap_statistics") or {})
            .get("median"),
            "alpha": queue_manifest.get("alpha", (
                config.get("queue_simulation") or {}).get("ALPHA")),
            "bpr_r2_median": r2.get("median"),
            "bpr_r2_min": r2.get("min"),
        })
    if allow_provenance and not records:
        records = _provenance_records(root)
        if records:
            _attach_root_bpr_r2(root, records)
    return records, all_statuses


def _attach_root_bpr_r2(root, records):
    """Fill BPR R^2 summaries from a root-level ``bpr_summary.csv`` export.

    Provenance directories carry no BPR manifests, but the archived bundle at
    the root level does.  Runs are matched on the network hash; the name prefix
    is used only when every bundle row of that prefix agrees on one calibration,
    because different revisions of an experiment can share a name but use
    different networks.
    """
    summary_path = root / "bpr_summary.csv"
    if not summary_path.is_file():
        return
    try:
        rows = list(csv.DictReader(summary_path.open()))
    except (OSError, csv.Error):
        return
    by_hash, by_prefix = {}, {}
    for row in rows:
        r2 = _read_json_value(row.get("r2_summary"))
        if r2 is None:
            continue
        network_hash = row.get("network_hash")
        if network_hash:
            by_hash.setdefault(network_hash, r2)
        prefix = str(row.get("run", "")).split("-")[0]
        by_prefix.setdefault(prefix, []).append(r2)
    for record in records:
        r2 = by_hash.get(record.get("network_hash"))
        if r2 is None:
            variants = by_prefix.get(record["name"].split("-")[0], [])
            distinct = {(entry.get("median"), entry.get("min")) for entry in variants}
            r2 = variants[0] if len(distinct) == 1 else None
        if r2 is not None:
            record["bpr_r2_median"] = r2.get("median")
            record["bpr_r2_min"] = r2.get("min")


def _read_json_value(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return {}


def _model_optimum(record, model):
    """Best evaluated objective of a run under one model.

    CG uses the minimum over every evaluated charger configuration (the
    exhaustive reference); queue uses the exhaustive strategy's objective,
    which equals the minimum over its evaluated placements.
    """
    if model == "cg":
        if record.get("cg_objectives"):
            return min(record["cg_objectives"])
        return _strategy_objective(record, "cg", "exhaustive")
    return _strategy_objective(record, "queue", "exhaustive")


def _strategy_objective(record, model, strategy):
    try:
        value = float(record[f"{model}_strategies"][strategy]["objective"])
    except (KeyError, TypeError, ValueError):
        return None
    return value


def _strategy_gap_pct(record, model, strategy):
    """Percent regret of a strategy against its run's best evaluated optimum."""
    objective = _strategy_objective(record, model, strategy)
    optimum = _model_optimum(record, model)
    if objective is None or optimum is None:
        return None
    return 100.0 * (objective - optimum) / optimum


def _placement_rows(selected):
    physical = [record for record in selected.values()
                if record["name"] in PHYSICAL_EXPERIMENTS]
    rows = []
    for strategy, label in REPORTING_STRATEGIES:
        cg = [_strategy_gap(record, "cg", strategy) for record in physical]
        q = [_strategy_gap(record, "queue", strategy) for record in physical]
        cg = [value for value in cg if value is not None]
        q = [value for value in q if value is not None]
        rows.append({
            "label": label,
            "runs": len(physical),
            "cg_exact": sum(value is not None
                            and abs(value) <= EXACT_TOLERANCE_PERCENT
                            for value in (_strategy_gap(record, "cg", strategy)
                                          for record in physical)),
            "queue_exact": sum(value is not None
                               and abs(value) <= EXACT_TOLERANCE_PERCENT
                               for value in (_strategy_gap(record, "queue", strategy)
                                             for record in physical)),
            "cg_median": statistics.median(cg) if cg else None,
            "cg_max": max(cg) if cg else None,
            "q_median": statistics.median(q) if q else None,
            "q_max": max(q) if q else None,
        })
    return rows, len(physical)


def _strategy_gap(record, model, strategy):
    return _strategy_gap_pct(record, model, strategy)


def _runtime_rows(records):
    """One row per distinct network: that network's calibration job.

    Runs are grouped by network hash so that different revisions sharing the
    same node/link counts are never averaged; per (nodes, edges) the dominant
    revision (most runs) is reported, and any competing revision gets its own
    hash-tagged row.  Within the winning group the job with the largest BPR
    time is the calibration job and supplies the reported stage times.
    """
    by_shape = {}
    for record in records:
        if record.get("nodes") is None or record.get("edges") is None:
            continue
        shape = (int(record["nodes"]), int(record["edges"]))
        by_shape.setdefault(shape, {}).setdefault(
            record.get("network_hash") or "", []).append(record)
    rows = []
    for (nodes, edges), revisions in sorted(by_shape.items()):
        dominant = max(revisions, key=lambda digest: len(revisions[digest]))
        pool = revisions[dominant]
        record = max(pool, key=lambda item: float(item.get("bpr_seconds") or 0.0))
        suffix = "" if len(revisions) == 1 else f" ({str(dominant)[:8]})"
        rows.append({
            "network": f"{nodes}/{edges}{suffix}",
            "vehicles": record.get("vehicles"),
            "bpr_minutes": (float(record["bpr_seconds"]) / 60
                            if record.get("bpr_seconds") is not None else None),
            "cg_minutes": (float(record["cg_seconds"]) / 60
                           if record.get("cg_seconds") is not None else None),
            "queue_minutes": (float(record["queue_ne_seconds"]) / 60
                              if record.get("queue_ne_seconds") is not None else None),
            "eval_minutes": (float(record["queue_comparison_seconds"]) / 60
                             if record.get("queue_comparison_seconds") is not None
                             else None),
            "total_minutes": (float(record["total_seconds"]) / 60
                              if record.get("total_seconds") is not None else None),
            "r2_median": record.get("bpr_r2_median"),
            "r2_min": record.get("bpr_r2_min"),
        })
    return rows


def _levels_label(labels):
    """Compact level list: slash-joined tokens, ellipsis for slash-bearing ones."""
    if not labels:
        return ""
    if any("/" in label for label in labels):
        if len(labels) >= 3:
            return f"$({labels[0]}\\ldots {labels[-1]})$"
        return "$(" + ",\\, ".join(labels) + ")$"
    return "/".join(labels)

def _queue_optimum(record):
    try:
        value = float(record["queue_strategies"]["exhaustive"]["objective"])
    except (KeyError, TypeError, ValueError):
        return None
    return value


def _robustness_rows(selected):
    """One row per solver-side factor, pooling the experiments that vary it.

    Every level solves the same physical problem.  Rows are skipped unless at
    least two of their levels are present in the cohort; the baseline is
    included in every pool (and therefore shared across rows, which the table
    notes disclose).  Per-level gap, iteration, and rho values are reported
    level by level, aligned with the printed levels.
    """
    rows = []
    for factor, levels in TABLE2_FACTOR_GROUPS:
        present = [(selected[experiment], label)
                   for experiment, label in levels if experiment in selected]
        if len(present) < 2:
            continue
        records = [record for record, _ in present]
        seen, labels = set(), []
        for _, label in present:
            if label not in seen:
                seen.add(label)
                labels.append(label)
        met = sum(record["converged"] for record in records)
        cycled = sum(record["cycles"] for record in records)
        searches = (met + cycled
                    + sum(record["nonconverged"] for record in records)
                    + sum(record["failed"] for record in records))

        def per_level(field, scale=1.0):
            values = []
            for record, _ in present:
                value = record.get(field)
                values.append(scale * float(value) if value is not None else None)
            return values

        agreements = [record for record in records
                      if record.get("top_1_agreement") is not None]
        rows.append({
            "factor": factor,
            "levels": _levels_label(labels),
            "met": met,
            "cycled": cycled,
            "searches": searches,
            "iter_levels": per_level("iterations_median"),
            "gap_levels": per_level("final_gap_median", 100.0),
            "rho_levels": per_level("spearman"),
            "top1": sum(bool(record["top_1_agreement"])
                        for record in agreements),
            "agreements": len(agreements),
        })
    return rows


def _problem_rows(selected):
    """One row per physical problem with its own optimal objectives.

    Objectives are also normalized to the baseline problem's optimum under
    the same model, so the baseline row reads 100 and the other rows show the
    relative congestion level of each scenario.
    """
    rows = []
    for label, experiment in TABLE3_PROBLEMS:
        record = selected.get(experiment)
        if record is None:
            continue
        queue_opt = _strategy_objective(record, "queue", "exhaustive")
        greedy_cg = _strategy_gap_pct(record, "cg", "greedy")
        greedy_q = _strategy_gap_pct(record, "queue", "greedy")
        met = record["converged"]
        cycled = record["cycles"]
        cg_minutes = (float(record["cg_seconds"]) / 60
                      if record.get("cg_seconds") is not None else None)
        q_minutes = (sum(float(record[field]) for field in
                         ("queue_ne_seconds", "queue_comparison_seconds")) / 60
                     if record.get("queue_ne_seconds") is not None
                     and record.get("queue_comparison_seconds") is not None
                     else None)
        rows.append({
            "label": label,
            "cg_optimum": _model_optimum(record, "cg"),
            "queue_optimum": queue_opt,
            "greedy_cg_gap": greedy_cg,
            "greedy_q_gap": greedy_q,
            "rho": record.get("spearman"),
            "same_best": record.get("top_1_agreement"),
            "met": met,
            "cycled": cycled,
            "searches": met + cycled + record["nonconverged"] + record["failed"],
            "cg_minutes": cg_minutes,
            "q_minutes": q_minutes,
        })
    base = next((row for row in rows if row["label"].startswith("Base:")), None)
    base_cg = base["cg_optimum"] if base else None
    base_q = base["queue_optimum"] if base else None
    for row in rows:
        row["cg_pct_base"] = (
            100.0 * row["cg_optimum"] / base_cg
            if base_cg and row["cg_optimum"] is not None else None)
        row["q_pct_base"] = (
            100.0 * row["queue_optimum"] / base_q
            if base_q and row["queue_optimum"] is not None else None)
    return rows


def _pooled_robustness(selected):
    met = sum(record["converged"] for record in selected.values())
    cycled = sum(record["cycles"] for record in selected.values())
    searches = (met + cycled
                + sum(record["nonconverged"] for record in selected.values())
                + sum(record["failed"] for record in selected.values()))
    rhos = [float(record["spearman"]) for record in selected.values()
            if record.get("spearman") is not None]
    pearsons = [float(record["pearson"]) for record in selected.values()
                if record.get("pearson") is not None]
    agreements = [record for record in selected.values()
                  if record.get("top_1_agreement") is not None]
    alphas = sorted({float(record["alpha"]) for record in selected.values()
                     if record.get("alpha") is not None
                     and math.isfinite(float(record["alpha"]))})
    return {
        "met": met,
        "cycled": cycled,
        "searches": searches,
        "rho_median": statistics.median(rhos) if rhos else None,
        "rho_min": min(rhos) if rhos else None,
        "rho_max": max(rhos) if rhos else None,
        "pearson_min": min(pearsons) if pearsons else None,
        "pearson_max": max(pearsons) if pearsons else None,
        "top1": sum(bool(record["top_1_agreement"]) for record in agreements),
        "agreements": len(agreements),
        "alphas": alphas,
    }


def _alpha_tex(alphas):
    """Format the convergence thresholds actually recorded by the runs."""
    if not alphas:
        return "the recorded $\\alpha$"
    percentages = [100.0 * value for value in alphas]
    if len(percentages) == 1:
        return f"$\\alpha={percentages[0]:g}\\%$"
    values = ",".join(f"{value:g}" for value in percentages)
    return f"$\\alpha\\in\\{{{values}\\}}\\%$"


def _latex_escape(value):
    """Escape characters that are special in LaTeX text mode."""
    return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_")


def _failed_note(all_statuses):
    failed = sorted({item["name"].split("-")[0] for item in all_statuses
                     if str(item.get("status", "")).startswith("failed")})
    if not failed:
        return ""
    return (" Failed jobs are excluded: "
            + ", ".join(_latex_escape(name) for name in failed) + ".")


def _placement_gap_sentence(rows):
    parts = []
    for row in rows:
        parts.append(
            f"{row['label']} {_short(row['cg_median'], 3)}/{_short(row['cg_max'], 3)}"
            f" (CG), {_short(row['q_median'], 3)}/{_short(row['q_max'], 3)} (Q)"
        )
    return "; ".join(parts)


def render_placement_table(rows, runtime, physical_runs, gap_sentence, failed_note):
    lines = [
        "% Generated by paper_tables.py; requires \\usepackage{booktabs}.",
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Greedy and baseline placement quality relative to exhaustive"
        " evaluation (top) and pipeline wall time (bottom). Placement is pooled"
        f" over {physical_runs} completed physical scenarios that vary the charger"
        " budget, demand level, charging share, or OD structure from the baseline.}",
        "  \\label{tab:reviewer_placement}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{3pt}",
        "  \\renewcommand{\\arraystretch}{1.10}",
        "  \\begin{tabular}{lc}",
        "    \\toprule",
        "    Strategy & Exact (CG/Q) \\\\",
        "    \\midrule",
    ]
    for row in rows:
        if row["label"] == "Random exp.":
            cell = "--"
        else:
            cell = (f"{row['cg_exact']}/{row['runs']} / "
                    f"{row['queue_exact']}/{row['runs']}")
        lines.append(f"    {row['label']} & {cell} \\\\")
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\vspace{4pt}",
        "",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{2.5pt}",
        "  \\begin{tabular}{lrcccccc}",
        "    \\toprule",
        "    N/L & Veh & BPR & CG & Q & Eval & Total &"
        " $R^2$ med/min \\\\",
        "    \\midrule",
    ])
    for row in runtime:
        r2 = (_short(row["r2_median"], 3) + "/" + _short(row["r2_min"], 3)
              if row["r2_median"] is not None and row["r2_min"] is not None else "--")
        vehicles = row["vehicles"] if row["vehicles"] is not None else "--"
        lines.append(
            f"    {row['network']} & {vehicles} & {_short(row['bpr_minutes'])} & "
            f"{_short(row['cg_minutes'])} & {_short(row['queue_minutes'])} & "
            f"{_short(row['eval_minutes'])} & {_short(row['total_minutes'])} & {r2} \\\\"
        )
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\vspace{2pt}",
        "  \\begin{minipage}{\\columnwidth}",
        "    \\scriptsize",
        "    \\textit{Notes:} Exact means numerical equality with the exhaustive"
        f" objective for the same scenario ($\\leq {EXACT_TOLERANCE_PERCENT:g}\\%$"
        " relative difference), not"
        " necessarily identical sites; the random row reports its expected"
        " objective. Regret is measured against the best evaluated objective of"
        " the same scenario and model (CG: minimum over every evaluated charger"
        " configuration). Median/max regret (\\% of the same-scenario exhaustive"
        f" objective): {gap_sentence}. Times are wall-clock minutes for each"
        " network's calibration job; the remaining jobs reuse that BPR"
        " calibration. $R^2$ is the in-sample coefficient of determination of the"
        " fitted BPR curve against simulated delay points, per link. The windows"
        " are nested DC extractions, not a controlled scaling experiment. Each"
        " queue cell is a cycle-state of the restricted-route better-response"
        " heuristic, not a verified Nash equilibrium."
        f"{failed_note}",
        "  \\end{minipage}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def render_robustness_table(rows, pooled, selected_count, failed_note):
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Queue better-response and cross-model diagnostics on the"
        " baseline problem. Each row pools the experiments that vary one stated"
        " solver factor; per-level values are listed in level order;"
        f" statistics use {selected_count} selected jobs.}}",
        "  \\label{tab:reviewer_robustness}",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{2pt}",
        "  \\renewcommand{\\arraystretch}{1.09}",
        "  \\begin{tabular}{llccccc}",
        "    \\toprule",
        "    Factor & Levels & Met/Cyc./Sets & Iters & Gap (\\%) & $\\rho$ \\\\",
        "    \\midrule",
    ]
    for row in rows:
        def level_cell(values, digits=2):
            cells = []
            for value in values:
                if value is None:
                    cells.append("--")
                elif digits == 0 or float(value).is_integer():
                    cells.append(str(int(round(value))))
                else:
                    cells.append(_short(value, digits))
            return "/".join(cells)
        lines.append(
            f"    {row['factor']} & {row['levels']} &"
            f" {row['met']}/{row['cycled']}/{row['searches']} &"
            f" {level_cell(row['iter_levels'], 0)} &"
            f" {level_cell(row['gap_levels'])} &"
            f" {level_cell(row['rho_levels'])} \\\\"
        )
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\vspace{2pt}",
        "  \\begin{minipage}{\\columnwidth}",
        "    \\scriptsize",
        "    \\textit{Notes:} ``Met'' means the restricted-route heuristic met "
        f"{_alpha_tex(pooled['alphas'])} minimum-cost-normalized gap test; cycled"
        " states are not Nash"
        " equilibria. Gap is the min-normalized final gap of one better-response"
        " search; $\\rho$ is the Spearman correlation between congestion-game"
        " and queue placement objectives over the evaluated charger sets;"
        " per-level values follow the listed levels (baseline first, where"
        " present). Factor rows pool their levels, including the baseline, so"
        " pooled counts share the baseline across rows. Across all jobs (each"
        f" counted once), {pooled['met']}/{pooled['searches']} searches met the"
        f" gap test and the top charger set matched in"
        f" {pooled['top1']}/{pooled['agreements']} jobs (median $\\rho$"
        f" {_short(pooled['rho_median'])}, range {_short(pooled['rho_min'])}--"
        f"{_short(pooled['rho_max'])}; Pearson {_short(pooled['pearson_min'])}--"
        f"{_short(pooled['pearson_max'])})."
        f"{failed_note}",
        "  \\end{minipage}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def render_problem_table(rows, failed_note, bpr_minutes=None):
    caption = (
        "\\caption{Per-problem results. Each row is one physical scenario that"
        " changes a problem-side factor (charger budget or candidate set,"
        " demand level, required-charging share, or OD structure) from the"
        " baseline; rows are self-contained and their objectives are not"
        " comparable across rows.")
    if bpr_minutes is not None:
        caption += (
            " The one-time BPR calibration took"
            f" {_fmt(bpr_minutes)} min and was reused by all other jobs.")
    caption += "}"
    lines = [
        "\\begin{table}[t]",
        "  \\centering",
        caption,
        "  \\label{tab:reviewer_problems}",
        "  \\scriptsize",
        "  \\setlength{\\tabcolsep}{2pt}",
        "  \\renewcommand{\\arraystretch}{1.09}",
        "  \\begin{tabular}{lccccrr}",
        "    \\toprule",
        "    Problem & Opt.\\ CG & Opt.\\ Q & Gr.\\ (CG/Q) & $\\rho$ &"
        " Met/Cyc./Sets & Time \\\\",
        "    \\midrule",
    ]
    for row in rows:
        cg_gap = _short(row["greedy_cg_gap"], 3) if row["greedy_cg_gap"] is not None else "--"
        q_gap = _short(row["greedy_q_gap"], 3) if row["greedy_q_gap"] is not None else "--"
        rho = _short(row["rho"]) if row["rho"] is not None else "--"
        if row.get("same_best"):
            rho += "$^\\star$"
        time_cell = (f"{_fmt(row['cg_minutes'])}/{_fmt(row['q_minutes'])}"
                     if row.get("cg_minutes") is not None
                     and row.get("q_minutes") is not None else "--")
        lines.append(
            f"    {row['label']} & {_short(row['cg_pct_base'])} &"
            f" {_short(row['q_pct_base'])} & {cg_gap}/{q_gap} & {rho} &"
            f" {row['met']}/{row['cycled']}/{row['searches']} & {time_cell} \\\\"
        )
    lines.extend([
        "    \\bottomrule",
        "  \\end{tabular}",
        "  \\vspace{2pt}",
        "  \\begin{minipage}{\\columnwidth}",
        "    \\scriptsize",
        "    \\textit{Notes:} Opt.\\ CG/Opt.\\ Q are the best evaluated optima of"
        " total travel time under the congestion game and the queue simulator,"
        " normalized to the baseline problem's optimum under the same model"
        " (baseline = 100); below 100 reflects the varied factor, and the"
        " 45-vehicle row scales with demand. $\\star$ marks scenarios whose"
        " best charger set is identical in both models. Greedy is the percent"
        " regret of greedy against its problem's best evaluated optimum (CG/Q);"
        " pipeline re-evaluations of the same CG configuration differ by up to"
        " 0.002\\%. Queue optima are best evaluated cycle-states, not verified"
        " Nash equilibria, with one final simulator realization per placement."
        " Time is stage wall-clock time in minutes (CG: congestion-game stage;"
        " Q: queue equilibrium search plus placement evaluation)."
        " All rows share the baseline scenario (90 vehicles, F2 share $2/3$,"
        " one OD pair, 6 candidates, $C=2$) except for the varied factor."
        " No zero-charger reference is part of the current suites;"
        " congestion-relief percentages relative to an uncharged baseline"
        " would require an additional run."
        f"{failed_note}",
        "  \\end{minipage}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


def build_tables(primary_root, runtime_roots=()):
    """Return the complete tex document (both tables) for the given roots."""
    records, all_statuses = _load_root_records(primary_root)
    selected = _latest_completed_records(records)
    if not selected:
        raise SystemExit(f"No completed runs found under {primary_root}")
    extra_records = []
    for root in runtime_roots:
        root_records, _ = _load_root_records(root, allow_provenance=True)
        extra_records.extend(record for record in root_records
                             if str(record.get("status", "")).startswith("complete"))
    placement, physical_runs = _placement_rows(selected)
    runtime = _runtime_rows(list(selected.values()) + extra_records)
    robustness = _robustness_rows(selected)
    pooled = _pooled_robustness(selected)
    failed = _failed_note(all_statuses)
    table1 = render_placement_table(
        placement, runtime, physical_runs,
        _placement_gap_sentence(placement), failed)
    table2 = render_robustness_table(robustness, pooled, len(selected), failed)
    problems = _problem_rows(selected)
    bpr_minutes = max(
        (float(record["bpr_seconds"]) / 60 for record in selected.values()
         if record.get("bpr_seconds") is not None),
        default=None)
    table3 = render_problem_table(problems, failed, bpr_minutes)
    header = (
        "% Reviewer-response tables generated by paper_tables.py\n"
        f"% Primary cohort: {primary_root}\n"
        f"% Runtime roots: {', '.join(runtime_roots) or primary_root}\n"
    )
    return header + table1 + "\n" + table2 + "\n" + table3


PREVIEW_TEMPLATE = r"""\documentclass[lettersize,journal]{IEEEtran}
\usepackage[T1]{fontenc}
\usepackage{booktabs}
\usepackage{url}
\begin{document}
\title{Reviewer-Response Tables}
\author{Generated by paper\_tables.py%
\thanks{Primary cohort: @@PRIMARY@@. Runtime roots: @@RUNTIME@@.}}
\maketitle
\noindent This preview contains only the proposed experimental tables.
It is not a new manuscript section.
@@TABLES@@
\end{document}
"""


def render_preview_document(table_tex, primary_root, runtime_roots):
    """Wrap the generated tables in a standalone IEEEtran preview document."""
    return (PREVIEW_TEMPLATE
            .replace("@@PRIMARY@@", _latex_escape(primary_root))
            .replace("@@RUNTIME@@",
                     _latex_escape(", ".join(runtime_roots) or primary_root))
            .replace("@@TABLES@@", "\n" + table_tex.rstrip() + "\n"))


def compile_pdf(tex_path, jobs=2):
    """Best-effort pdflatex compilation next to ``tex_path``.

    Returns the PDF path on success, or None when pdflatex is unavailable or
    the document fails to compile (the tex file is left in place either way).
    """
    tex_path = Path(tex_path)
    if shutil.which("pdflatex") is None:
        print("pdflatex not found; wrote the tex only")
        return None
    ok = True
    for _ in range(jobs):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", tex_path.name],
            cwd=tex_path.parent,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        ok = ok and result.returncode == 0
    pdf = tex_path.with_suffix(".pdf")
    if ok and pdf.is_file():
        return pdf
    print(f"pdflatex failed; inspect the log at {tex_path.with_suffix('.log')}")
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate the two reviewer-response LaTeX tables.")
    parser.add_argument("--results-root", action="append", required=True,
                        help="results folder (repeatable; first is the primary cohort)")
    parser.add_argument("--runtime-root", action="append", default=[],
                        help="extra results folder contributing runtime rows only")
    parser.add_argument("--output", default=None,
                        help="output .tex path (default: <primary root>/reviewer_tables.tex)")
    parser.add_argument("--preview", action="store_true",
                        help="also write a standalone IEEEtran preview tex "
                             "next to the output")
    parser.add_argument("--compile", action="store_true",
                        help="compile the preview tex with pdflatex (requires --preview)")
    parser.add_argument("--stdout", action="store_true",
                        help="also print the generated tex")
    args = parser.parse_args()

    document = build_tables(args.results_root[0], args.runtime_root)
    output = Path(args.output) if args.output else (
        Path(args.results_root[0]) / "reviewer_tables.tex")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)
    if args.stdout:
        print(document)
    print(f"Wrote {output}")

    if args.preview:
        preview = output.with_name(
            output.stem.replace("_tables", "_tables_preview") + ".tex")
        preview.write_text(render_preview_document(
            document, args.results_root[0], args.runtime_root))
        print(f"Wrote {preview}")
        if args.compile:
            pdf = compile_pdf(preview)
            if pdf is not None:
                print(f"Compiled {pdf}")


if __name__ == "__main__":
    main()
