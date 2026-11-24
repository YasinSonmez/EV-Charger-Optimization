#!/usr/bin/env python3
"""Regenerate the objective-comparison figure (Fig. 10) for a completed run
from its saved summary, without re-running any experiment.

Usage:
    python scripts/plot_objective_comparison.py \
        [--run-dir results/suite_new6_500/runs/sensitivity_base-ff8efbae0bc5] \
        [--images-dir "docs/.../images"] [--images-name 10_objective_comparison.png]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",
                        default="results/suite_new6_500/runs/sensitivity_base-ff8efbae0bc5")
    parser.add_argument("--images-dir",
                        default="docs/Optimum_EV_Placement__Journal_ (3)/images")
    parser.add_argument("--images-name", default="10_objective_comparison.png")
    args = parser.parse_args()

    from pipeline import _plot_objective_comparison

    with open(os.path.join(args.run_dir, "experiment_summary.json")) as handle:
        summary = json.load(handle)
    cg_results = summary.get("cg_results") or {}
    queue_results = summary.get("queue_results") or {}

    candidate_labels = None
    resolved_path = os.path.join(args.run_dir, "resolved_config.json")
    if os.path.isfile(resolved_path):
        with open(resolved_path) as handle:
            resolved = json.load(handle)
        scenario = resolved.get("generated_scenario") or {}
        if scenario.get("candidates"):
            candidate_labels = {
                int(candidate["node_id"]): f"C{index + 1}"
                for index, candidate in enumerate(scenario["candidates"])
            }

    run_output = os.path.join(args.run_dir, "plots", "objective_comparison.png")
    _plot_objective_comparison(cg_results, queue_results, run_output,
                               candidate_labels=candidate_labels)
    images_output = os.path.join(args.images_dir, args.images_name)
    _plot_objective_comparison(cg_results, queue_results, images_output,
                               candidate_labels=candidate_labels)
    print(f"wrote {run_output}")
    print(f"wrote {images_output}")


if __name__ == "__main__":
    main()
