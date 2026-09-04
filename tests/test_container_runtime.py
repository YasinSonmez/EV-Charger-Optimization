"""Small contracts used by Docker and Apptainer execution."""

import json
import pickle

import numpy as np
import pandas as pd

import pipeline
from src.contracts import SeedManager
from src.network_artifact import write_network_artifact
from src.run_state import process_provenance


class _CachedFitter:
    def __init__(self, frame):
        self.df = frame
        self.fit_metadata = {"source": "test-cache"}

    def save_results_to_csv(self, path):
        self.df.to_csv(path, index=False)


def test_explicit_liblsp_path_has_priority(monkeypatch, tmp_path):
    from queue_sim.interface import _find_library

    library = tmp_path / "liblsp.so"
    library.touch()
    monkeypatch.setenv("EVOPT_LIBLSP_PATH", str(library))
    assert _find_library() == str(library.resolve())


def test_explicit_liblsp_path_must_exist(monkeypatch, tmp_path):
    from queue_sim.interface import _find_library

    missing = tmp_path / "missing.so"
    monkeypatch.setenv("EVOPT_LIBLSP_PATH", str(missing))
    try:
        _find_library()
    except OSError as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("missing EVOPT_LIBLSP_PATH was accepted")


def test_container_provenance_is_recorded(monkeypatch):
    monkeypatch.setenv("EVOPT_CONTAINERIZED", "1")
    monkeypatch.setenv("EVOPT_IMAGE_REF", "example/evopt:test")
    monkeypatch.setenv("EVOPT_IMAGE_DIGEST", "sha256:123")
    monkeypatch.setenv("EVOPT_EXECUTION_MODE", "workspace")
    monkeypatch.setenv("EVOPT_CODE_COMMIT", "abc123")

    provenance = process_provenance()
    assert provenance["containerized"] is True
    assert provenance["container_image"] == "example/evopt:test"
    assert provenance["container_image_digest"] == "sha256:123"
    assert provenance["execution_mode"] == "workspace"
    assert provenance["code_commit"] == "abc123"


def test_bpr_checkpoint_resume_uses_module_pandas(tmp_path):
    artifact = tmp_path / "network"
    manifest = write_network_artifact(
        pd.DataFrame({"node_id": [0, 1], "lon": [0.0, 1.0], "lat": [0.0, 0.0]}),
        pd.DataFrame({
            "link_id": [0], "start_node_id": [0], "end_node_id": [1],
            "edge_key": [0], "length": [100.0],
        }),
        artifact,
    )
    frame = pd.DataFrame({
        "link_id": [0], "x_vector": [np.array([0.0, 1.0])],
        "y_vector": [np.array([10.0, 11.0])], "a_fit": [0.15],
        "b_fit": [4.0], "cap_fit": [1000.0], "fft_fit": [10.0],
        "fit_status": ["full"], "network_hash": [manifest["network_hash"]],
    })
    work_dir = tmp_path / "bpr"
    work_dir.mkdir()
    cache_path = work_dir / "cached_results.pkl"
    with cache_path.open("wb") as handle:
        pickle.dump((frame, _CachedFitter(frame)), handle)
    (work_dir / "bpr_manifest.json").write_text(json.dumps({
        "network_hash": manifest["network_hash"],
        "bpr_mode": "historical_artifact_compatible", "num_samples": 25,
        "max_flow": 250.0, "random_seed": 0, "fitter_version": "historical_v1",
        "route_semantics": "measured_target_flow_with_straight_ahead_context",
        "fit_screening": "none", "correlation_threshold": 0.0,
        "variation_ratio_threshold": 0.0, "accept_low_r2": True,
        "missing_context_policy": "synthetic_boundary",
        "synthetic_context_capacity_multiplier": 10.0,
        "synthetic_context_length_m": 1.0, "simulation_horizon": 10801,
    }))

    resumed, _ = pipeline.load_or_fit_model(
        data_path=str(work_dir / "missing.csv"), cache_path=str(cache_path),
        work_dir=str(work_dir), artifact_dir=str(artifact), n_links=1,
        seed_manager=SeedManager(0),
    )
    assert resumed["fit_status"].tolist() == ["full"]
    assert resumed["link_length"].tolist() == [100.0]
