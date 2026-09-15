"""Contracts for the final reviewer-driven manifests."""

import json
from pathlib import Path
import zipfile

import pytest

from run_suite import (
    REPORTING_STRATEGIES,
    SENSITIVITY_FACTORS,
    SENSITIVITY_PHYSICAL_EXPERIMENTS,
    _latest_completed_records,
    _render_sensitivity_table_tex,
    _sensitivity_paper_rows,
    export_bundle,
    load_manifest,
    run_jobs,
    select_jobs,
)
from src.config import Config


ROOT = Path(__file__).resolve().parents[1]


def test_start_index_selects_suffix_and_index_selects_one():
    jobs = [{"id": f"job_{index}"} for index in range(4)]
    assert [job["id"] for job in select_jobs(jobs, index=2)] == ["job_2"]
    assert [job["id"] for job in select_jobs(jobs, start_index=2)] == [
        "job_2", "job_3",
    ]
    assert [job["id"] for job in select_jobs(jobs)] == [
        "job_0", "job_1", "job_2", "job_3",
    ]


def test_start_index_rejects_conflict_and_out_of_range():
    jobs = [{"id": "job_0"}]
    try:
        select_jobs(jobs, index=0, start_index=0)
    except ValueError as exc:
        assert "cannot be used together" in str(exc)
    else:
        raise AssertionError("conflicting index options were accepted")
    try:
        select_jobs(jobs, start_index=2)
    except IndexError:
        pass
    else:
        raise AssertionError("out-of-range start index was accepted")


def test_continue_on_failure_records_failure_and_runs_later_jobs(
    monkeypatch, tmp_path,
):
    _manifest, jobs = load_manifest(
        ROOT / "configs/rebuttal/smoke/smoke_suite.json",
        expanded_dir=tmp_path / "expanded",
    )
    attempted = []

    def fake_run_job(job, _jobs_by_id, _results_root, _resume):
        attempted.append(job["id"])
        if job["id"] == "smoke_base":
            raise RuntimeError("intentional test failure")

    monkeypatch.setattr("run_suite.run_job", fake_run_job)
    failures = run_jobs(
        jobs, {job["id"]: job for job in jobs}, tmp_path / "results", False,
        continue_on_failure=True,
    )

    assert attempted == ["smoke_base", "smoke_cg_routes"]
    assert [job_id for job_id, _ in failures] == ["smoke_base"]


def test_final_suite_sizes_and_fixed_alpha(tmp_path):
    _, scaling = load_manifest(
        ROOT / "configs/rebuttal/final/scaling_suite.json", tmp_path / "scaling"
    )
    _, sensitivity = load_manifest(
        ROOT / "configs/rebuttal/final/sensitivity_suite.json", tmp_path / "sensitivity"
    )
    assert len(scaling) == 3
    assert len(sensitivity) == 20
    assert [job["id"] for job in sensitivity[:6]] == [
        "sensitivity_base", "budget_6_3", "budget_6_4",
        "budget_7_2", "budget_7_3", "budget_7_4",
    ]
    assert "route_source_independent" in {job["id"] for job in sensitivity}
    assert "init_uniform" in {job["id"] for job in sensitivity}
    assert "route_source_cg" not in {job["id"] for job in sensitivity}
    assert "init_shortest" not in {job["id"] for job in sensitivity}
    assert all(Config.from_dict(job["raw"]).queue_simulation["ALPHA"] == 0.01
               for job in sensitivity)


def test_sensitivity_dependencies_and_bpr_seed_are_immutable(tmp_path):
    _, jobs = load_manifest(
        ROOT / "configs/rebuttal/final/sensitivity_suite.json", tmp_path
    )
    for job in jobs[1:]:
        assert job["reuse_network_from"] == "sensitivity_base"
        assert job["reuse_bpr_from"] == "sensitivity_base"
    assert {
        Config.from_dict(job["raw"]).pipeline["bpr_generation"]["seed"]
        for job in jobs
    } == {42}
    assert {
        Config.from_dict(job["raw"]).queue_simulation["seed"]
        for job in jobs if job["id"].startswith("queue_seed_")
    } == {43, 44}


def test_balanced_route_settings_are_enabled_in_final_jobs(tmp_path):
    _, jobs = load_manifest(
        ROOT / "configs/rebuttal/final/sensitivity_suite.json", tmp_path
    )
    for job in jobs:
        queue = Config.from_dict(job["raw"]).queue_simulation
        assert queue["balanced_charger_routes"] is True
        assert queue["K"] >= Config.from_dict(job["raw"]).num_chargers


def test_primary_jobs_use_paper_route_source_and_initialization(tmp_path):
    _, scaling = load_manifest(
        ROOT / "configs/rebuttal/final/scaling_suite.json", tmp_path / "scaling"
    )
    _, sensitivity = load_manifest(
        ROOT / "configs/rebuttal/final/sensitivity_suite.json", tmp_path / "sensitivity"
    )
    primary = scaling + [job for job in sensitivity if job["id"] == "sensitivity_base"]
    for job in primary:
        queue = Config.from_dict(job["raw"]).queue_simulation
        assert queue["route_source"] == "cg_recovered_top_k"
        assert queue["initialization"] == "cg_proportional"


def test_reporting_cohort_uses_latest_completed_digest():
    records = [
        {"name": "sensitivity_base", "run": "base-old", "status": "complete",
         "timestamp_utc": "2026-01-01T00:00:00Z"},
        {"name": "sensitivity_base", "run": "base-failed", "status": "failed",
         "timestamp_utc": "2026-01-03T00:00:00Z"},
        {"name": "sensitivity_base", "run": "base-new", "status": "complete_with_approximate_ne",
         "timestamp_utc": "2026-01-02T00:00:00Z"},
        {"name": "not_in_manifest", "run": "extra", "status": "complete",
         "timestamp_utc": "2026-01-04T00:00:00Z"},
    ]

    selected = _latest_completed_records(records, ["sensitivity_base"])

    assert selected["sensitivity_base"]["run"] == "base-new"
    assert "not_in_manifest" not in selected


def _paper_record(name, index):
    physical_index = (
        SENSITIVITY_PHYSICAL_EXPERIMENTS.index(name)
        if name in SENSITIVITY_PHYSICAL_EXPERIMENTS else None
    )
    cg_limits = {"greedy": 10, "single_swap": 11,
                 "minimum_detour": 4, "weighted_betweenness": 3,
                 "uniform_random_expectation": 0}
    queue_limits = {"greedy": 7, "single_swap": 9,
                    "minimum_detour": 0, "weighted_betweenness": 0,
                    "uniform_random_expectation": 0}

    def strategies(limits):
        return {
            strategy: {"regret_pct": 0.0 if physical_index is not None
                       and physical_index < limits[strategy] else 1.0}
            for strategy, _label in REPORTING_STRATEGIES
        }

    return {
        "name": name, "run": f"{name}-digest", "status": "complete_with_approximate_ne",
        "timestamp_utc": "2026-01-01T00:00:00Z", "config_digest": f"digest-{index}",
        "cg_strategies": strategies(cg_limits),
        "queue_strategies": strategies(queue_limits),
        "spearman": -0.3 + index * 0.06, "top_1_agreement": index < 5,
        "nodes": 106, "edges": 229, "cycles": 30, "converged": 0,
        "nonconverged": 0, "failed": 0, "bpr_seconds": 1.0,
        "cg_seconds": 120.0, "queue_ne_seconds": 300.0,
        "queue_comparison_seconds": 180.0, "other_seconds": 19.0,
        "total_seconds": 620.0,
    }


def test_horizontal_sensitivity_table_contract():
    experiment_names = []
    for _factor, _levels, names in SENSITIVITY_FACTORS:
        for name in names:
            if name not in experiment_names:
                experiment_names.append(name)
    selected = {
        name: _paper_record(name, index)
        for index, name in enumerate(experiment_names)
    }
    assert len(selected) == 20

    rows = _sensitivity_paper_rows(selected)
    placement = {row["factor"]: row for row in rows
                 if row["panel"] == "A_placement"}
    assert placement["Greedy"]["cg_exact"] == 10
    assert placement["Greedy"]["queue_exact"] == 7
    assert placement["Greedy + swap"]["cg_exact"] == 11
    assert placement["Greedy + swap"]["queue_exact"] == 9

    agreement = {row["factor"]: row for row in rows
                 if row["panel"] == "B_agreement"}
    assert agreement["Route src."]["runs"] == 2
    assert agreement["Init."]["runs"] == 3
    assert agreement["Overall"]["top_1_agreement_count"] == 5
    assert agreement["Overall"]["mean_bpr_minutes"] == pytest.approx(1 / 20 / 60)
    assert agreement["Overall"]["mean_cg_minutes"] == pytest.approx(2)
    assert agreement["Overall"]["mean_queue_minutes"] == pytest.approx(5)
    assert agreement["Overall"]["mean_total_minutes"] == pytest.approx(620 / 60)

    tex = _render_sensitivity_table_tex(rows, selected)
    assert "Panel A:" not in tex and "Panel B:" not in tex and "Panel C:" not in tex
    assert "\\normalsize" in tex and "\\scriptsize" not in tex
    assert "BPR/CG/Q/Total" in tex
    assert "Jobs &" not in tex
    assert "Median regret" not in tex
    assert "Median-iteration" not in tex
    assert "Cycles &" not in tex
    assert "Q is the queue cycle-state approximation" in tex


def test_export_bundle_emits_tex_and_selected_digest(monkeypatch, tmp_path):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    root = tmp_path / "results"
    run_dir = root / "sensitivity_base-newdigest"
    (run_dir / "queue").mkdir(parents=True)
    status = {
        "status": "complete_with_approximate_ne", "eligible": False,
        "config_name": "sensitivity_base", "config_digest": "newdigest",
        "timestamp_utc": "2026-01-02T00:00:00Z",
    }
    (run_dir / "status.json").write_text(json.dumps(status))
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "network": {"node_count": 106, "edge_count": 229},
        "timing": {"total": 620, "bpr_fitting": 1, "cg_optimization": 120,
                   "queue_ne": 300, "queue_comparison": 180},
    }))
    strategies = {
        strategy: {"placement": [1, 2], "objective": 10.0,
                   "regret_pct": 0.0 if strategy != "uniform_random_expectation" else 1.0}
        for strategy, _label in REPORTING_STRATEGIES
    }
    summary = {
        "config": {"name": "sensitivity_base",
                   "scenario_generation": {"demand": {"F1": 60, "F2": 120}}},
        "cg_results": {"num_configs": 15, "all_configs": []},
        "queue_results": {
            "reviewer_baselines": {
                "congestion_game": strategies, "queue": strategies,
            },
            "ne_statistics": {"status_counts": {"cycle": 15}},
            "cg_queue_correlations": {
                "spearman": 0.5, "pearson": 0.6, "top_1_agreement": True,
            },
            "exhaustive_results": [],
        },
    }
    (run_dir / "experiment_summary.json").write_text(json.dumps(summary))
    (run_dir / "queue" / "queue_manifest.json").write_text(json.dumps({
        "status_counts": {"cycle": 15},
    }))

    destination = tmp_path / "bundle.zip"
    export_bundle(root, destination, experiment_ids=["sensitivity_base"])

    with zipfile.ZipFile(destination) as archive:
        names = set(archive.namelist())
        assert "rebuttal_sensitivity_table.tex" in names
        cohort = json.loads(archive.read("reporting_cohort.json"))
        assert cohort["sensitivity_base"]["config_digest"] == "newdigest"
        tex = archive.read("rebuttal_sensitivity_table.tex").decode()
        assert "sensitivity statistics use 1 selected jobs" in tex
        paper_csv = archive.read("paper_table.csv").decode().splitlines()[0]
        assert "cg_exact" in paper_csv and "mean_queue_minutes" in paper_csv
        assert "median_greedy_gap" not in paper_csv and "cycle_rate" not in paper_csv
