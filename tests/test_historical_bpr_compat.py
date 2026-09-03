"""Regression tests for the historical BPR compatibility contract."""

from __future__ import annotations

import io
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import pipeline
from src.config import Config
from src.model_fitter import TrafficModelFitter, convert_string_to_array, model


def _git_file(path: str) -> bytes:
    try:
        return subprocess.check_output(
            ["git", "show", f"37eab33:{path}"], stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"historical git fixture unavailable: {exc}")


def test_historical_reference_has_122_links_and_25_samples():
    traffic = pd.read_csv(io.BytesIO(_git_file("data/traffic_data.csv")))
    assert len(traffic) == 122
    convert_string_to_array(traffic, "x_vector")
    convert_string_to_array(traffic, "y_vector")
    assert {len(values) for values in traffic["x_vector"]} == {25}
    assert {len(values) for values in traffic["y_vector"]} == {25}


def test_historical_fitter_reproduces_reference_behavior(tmp_path):
    traffic = pd.read_csv(io.BytesIO(_git_file("data/traffic_data.csv")))
    expected = pd.read_csv(io.BytesIO(_git_file("data/fitter_results.csv")))
    convert_string_to_array(traffic, "x_vector")
    convert_string_to_array(traffic, "y_vector")

    fitter = TrafficModelFitter(pandas_df=traffic)
    fitter.parallel_fit_and_evaluate(
        workers=1,
        output_dir=tmp_path,
        save_plots=False,
        fit_mode="historical_artifact_compatible",
        validation_mode="parameter_complete",
    )
    actual = fitter.df.sort_values("link_id").reset_index(drop=True)
    expected = expected.sort_values("link_id").reset_index(drop=True)

    # The four-parameter nonlinear problem is weakly identified for some
    # nearly-flat links, so compare parameters at a rough-equivalence
    # tolerance and compare the resulting curves much more tightly.
    for column in ("a_fit", "b_fit", "cap_fit", "fft_fit"):
        assert np.allclose(
            actual[column].astype(float), expected[column].astype(float),
            rtol=5e-2, atol=1e-3, equal_nan=True,
        )
    assert np.allclose(
        actual["R^2"].astype(float), expected["R^2"].astype(float),
        rtol=1e-5, atol=1e-6, equal_nan=True,
    )
    assert int((actual["fit_status"] == "constant_fallback").sum()) == 6
    assert "honest_R2" in actual.columns


def test_historical_constant_fallback_reports_honest_r2(tmp_path):
    frame = pd.DataFrame([{
        "link_id": 0,
        "x_vector": np.arange(1, 6, dtype=float),
        "y_vector": np.array([10.0, 10.2, 9.8, 10.1, 9.9]),
    }])
    fitter = TrafficModelFitter(pandas_df=frame)
    fitter.parallel_fit_and_evaluate(
        workers=1,
        output_dir=tmp_path,
        save_plots=False,
        fit_mode="historical_artifact_compatible",
        validation_mode="parameter_complete",
    )
    row = fitter.df.iloc[0]
    assert row["fit_status"] == "constant_fallback"
    assert row["R^2"] == pytest.approx(1.0)
    assert row["honest_R2"] < 1.0


def test_proxy_source_is_not_reclassified_as_a_measured_fit(tmp_path):
    frame = pd.DataFrame([{
        "link_id": 0,
        "x_vector": np.arange(1, 6, dtype=float),
        "y_vector": np.array([10.0] * 5),
        "fit_status": "proxy",
        "observation_source": "proxy",
    }])
    fitter = TrafficModelFitter(pandas_df=frame)
    fitter.parallel_fit_and_evaluate(
        workers=1,
        output_dir=tmp_path,
        save_plots=False,
        fit_mode="historical_artifact_compatible",
        validation_mode="parameter_complete",
        fit_screening="none",
        accept_low_r2=True,
    )
    row = fitter.df.iloc[0]
    assert row["observation_source"] == "proxy"
    assert row["fit_status"] == "proxy"


def test_strict_capacity_fraction_mode_remains_explicit():
    raw = {
        "coordinates": [1, 0, 1, 0],
        "num_chargers": 1,
        "possible_charger_positions": [1],
        "od_demand": {"0,1": [1, 1]},
        "pipeline": {"bpr_generation": {"mode": "capacity_fraction_strict"}},
    }
    config = Config.from_dict(raw)
    bpr = config.pipeline["bpr_generation"]
    assert bpr["mode"] == "capacity_fraction_strict"
    assert bpr["fit_validation"] == "full"
    assert bpr["fixed_references"] is True


def test_relaxed_historical_fit_can_retain_a_finite_low_quality_curve(tmp_path):
    # Deliberately noisy observations exercise the relaxed path.  The fit is
    # retained for inspection, while its honest R² remains low and visible.
    x = np.arange(1, 26, dtype=float)
    y = np.array([
        10.0, 12.0, 9.0, 11.5, 8.5, 13.0, 9.5, 10.5, 8.0, 12.5,
        9.0, 11.0, 8.5, 13.5, 9.5, 10.0, 8.0, 12.0, 9.0, 11.5,
        8.5, 13.0, 9.5, 10.5, 8.0,
    ])
    frame = pd.DataFrame([{"link_id": 0, "x_vector": x, "y_vector": y}])
    fitter = TrafficModelFitter(pandas_df=frame)
    fitter.parallel_fit_and_evaluate(
        workers=1,
        output_dir=tmp_path,
        save_plots=False,
        fit_mode="historical_artifact_compatible",
        validation_mode="parameter_complete",
        r2_threshold=0.5,
        fit_screening="none",
        accept_low_r2=True,
    )
    row = fitter.df.iloc[0]
    assert row["fit_status"] == "full_relaxed"
    assert np.isfinite(row[["a_fit", "b_fit", "cap_fit", "fft_fit", "honest_R2"]].astype(float)).all()
    assert row["honest_R2"] < 0.5


def test_config_exposes_relaxed_historical_controls():
    raw = {
        "coordinates": [1, 0, 1, 0],
        "num_chargers": 1,
        "possible_charger_positions": [1],
        "od_demand": {"0,1": [1, 1]},
        "pipeline": {"bpr_generation": {
            "fit_screening": "none",
            "correlation_threshold": 0.0,
            "variation_ratio_threshold": 0.0,
            "accept_low_r2": True,
        }},
    }
    bpr = Config.from_dict(raw).pipeline["bpr_generation"]
    assert bpr["fit_screening"] == "none"
    assert bpr["correlation_threshold"] == 0.0
    assert bpr["variation_ratio_threshold"] == 0.0
    assert bpr["accept_low_r2"] is True


def test_bpr_manifest_identity_rejects_stale_historical_data(tmp_path):
    manifest_path = tmp_path / "bpr_manifest.json"
    manifest_path.write_text(
        '{"network_hash":"network-a","bpr_mode":"historical_artifact_compatible",'
            '"num_samples":25,"max_flow":250,"random_seed":42,'
            '"fitter_version":"historical_v1",'
            '"route_semantics":"measured_target_flow_with_straight_ahead_context",'
            '"fit_screening":"legacy","correlation_threshold":0.3,'
            '"variation_ratio_threshold":0.03,"accept_low_r2":false,'
            '"missing_context_policy":"synthetic_boundary",'
            '"synthetic_context_capacity_multiplier":10.0,'
            '"synthetic_context_length_m":1.0,"simulation_horizon":10801}'
    )
    request = {
        "mode": "historical_artifact_compatible",
        "num_samples": 25,
        "max_flow": 250,
        "fit_screening": "legacy",
        "correlation_threshold": 0.3,
        "variation_ratio_threshold": 0.03,
        "accept_low_r2": False,
        "missing_context_policy": "synthetic_boundary",
        "synthetic_context_capacity_multiplier": 10.0,
        "synthetic_context_length_m": 1.0,
        "simulation_horizon": 10801,
    }
    assert pipeline._bpr_manifest_is_compatible(str(manifest_path), "network-a", request, 42)
    assert not pipeline._bpr_manifest_is_compatible(str(manifest_path), "network-b", request, 42)
    request["mode"] = "capacity_fraction_strict"
    assert not pipeline._bpr_manifest_is_compatible(str(manifest_path), "network-a", request, 42)


def test_historical_worker_uses_measured_target_flow(monkeypatch, tmp_path):
    import queue_sim.bpr_data_generator as generator

    target = {
        "link_id": 1,
        "start_node_id": 1,
        "end_node_id": 2,
        "length": 100.0,
        "maxmph": 25.0,
        "lanes": 1,
    }
    predecessor = pd.Series({"link_id": 0, "start_node_id": 0, "end_node_id": 1})
    successor = pd.Series({"link_id": 2, "start_node_id": 2, "end_node_id": 3})

    class FakeRunner:
        def __init__(self, **kwargs):
            self.nodes_df = pd.DataFrame({
                "node_id": [0, 1, 2, 3],
                "lat": [0.0] * 4,
                "lon": [0.0, 1.0, 2.0, 3.0],
            })
            self.links_df = pd.DataFrame([
                {"link_id": 0, "start_node_id": 0, "end_node_id": 1},
                target,
                {"link_id": 2, "start_node_id": 2, "end_node_id": 3},
            ])
            self.od_df = pd.DataFrame()
            self.sim = SimpleNamespace(all_links={})

        def find_sa_in_link(self, link):
            return predecessor

        def find_sa_out_link(self, link):
            return successor

        def add_charging_info(self, *_args):
            return None

        def init_sq_simulation_for_bpr_function_fitting_V2(self, link, *_args):
            demand = len(self.od_df)
            self.sim.all_links[1] = SimpleNamespace(
                completed_travel_time_list=[12.0] * demand,
                tot_entering_vehs=demand,
                ave_travel_time=12.0,
                queue_veh=[],
                run_veh=[],
            )

        def spatial_queue_simulation(self, *_args, **_kwargs):
            return None

        def return_traffic_data(self, *_args, **_kwargs):
            return pd.DataFrame({
                "link_id": [1],
                "flow": [17.5],
                "travel_time": [12.0],
            })

    monkeypatch.setattr(generator, "Runner", FakeRunner)
    result = generator._bpr_link_worker((
        target,
        str(tmp_path / "nodes.csv"),
        str(tmp_path / "edges.csv"),
        {
            "mode": "historical_artifact_compatible",
            "flow_levels": [1, 2],
            "route_mode": "contextual",
        },
        str(tmp_path / "sweeps"),
        42,
    ))
    assert result["errors"] == []
    assert result["x_vector"] == [17.5, 17.5]
    assert result["y_vector"] == [12.0, 12.0]
