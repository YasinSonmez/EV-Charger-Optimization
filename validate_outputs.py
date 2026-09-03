#!/usr/bin/env python3
"""Validate a completed experiment directory.

Examples:
    python validate_outputs.py results/<experiment>
    python validate_outputs.py results/<experiment> --queue
    python validate_outputs.py results/<network-only-run> --network-only
"""

from __future__ import annotations

import argparse
import json

from src.sanity_checks import validate_experiment_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate EV optimization output artifacts")
    parser.add_argument("experiment_dir")
    parser.add_argument("--queue", action="store_true", help="require queue NE/comparison artifacts")
    parser.add_argument("--network-only", action="store_true", help="validate a network-only run")
    args = parser.parse_args()

    result = validate_experiment_outputs(
        args.experiment_dir,
        require_cg=not args.network_only,
        require_queue=args.queue,
        require_reports=not args.network_only,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
