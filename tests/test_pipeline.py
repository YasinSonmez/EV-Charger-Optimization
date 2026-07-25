"""Integration test: full pipeline with small parameters.

Run with: pytest tests/test_pipeline.py -v --tb=short -s
Or directly: python tests/test_pipeline.py
"""
import os
import sys
import shutil
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.mark.timeout(600)
def test_pipeline_end_to_end():
    """Run the full pipeline with config_test.json (small params) and verify outputs."""
    from pipeline import run_pipeline

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(project_root, "config_test.json")

    if not os.path.exists(config_path):
        pytest.skip(f"Test config not found: {config_path}")

    experiment_dir = run_pipeline(config_path)

    try:
        assert os.path.isdir(experiment_dir), f"Experiment dir not created: {experiment_dir}"
        assert os.path.exists(os.path.join(experiment_dir, "run_config.json")), "run_config.json missing"
        assert os.path.exists(os.path.join(experiment_dir, "all_optimization_results.pkl")), "CG pkl missing"
        assert os.path.exists(os.path.join(experiment_dir, "report.md")), "report.md missing"

        from queue_sim import QUEUE_SIM_AVAILABLE
        if QUEUE_SIM_AVAILABLE:
            queue_dir = os.path.join(experiment_dir, "queue")
            assert os.path.isdir(queue_dir), "queue/ dir missing"
            assert os.path.exists(os.path.join(queue_dir, "NE_path_assignments.pkl")), "NE pkl missing"
            assert os.path.exists(os.path.join(queue_dir, "comparison_results.json")), "comparison results missing"

            import json
            with open(os.path.join(queue_dir, "comparison_results.json")) as f:
                results = json.load(f)
            assert "best_greedy" in results
            assert "best_exhaustive" in results
            assert "suboptimality_pct" in results
            assert isinstance(results["suboptimality_pct"], (int, float))
    finally:
        if os.path.isdir(experiment_dir):
            shutil.rmtree(experiment_dir, ignore_errors=True)


if __name__ == "__main__":
    test_pipeline_end_to_end()
    print("Integration test PASSED")
