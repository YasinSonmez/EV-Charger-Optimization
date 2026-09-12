"""Contracts for the final reviewer-driven manifests."""

from pathlib import Path

from run_suite import load_manifest, run_jobs, select_jobs
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
