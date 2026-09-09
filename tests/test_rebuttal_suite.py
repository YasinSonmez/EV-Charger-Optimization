"""Contracts for the final reviewer-driven manifests."""

from pathlib import Path

from run_suite import load_manifest
from src.config import Config


ROOT = Path(__file__).resolve().parents[1]


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
