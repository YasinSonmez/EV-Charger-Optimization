"""Machine-checkable validation of experiment artifacts.

The pipeline produces several files that are consumed by later stages.  A
successful Python process is not enough evidence that those files describe
the same experiment: a stale network, an incomplete BPR table, or a partially
written queue result can all look plausible.  This module validates the
cross-file invariants that make an experiment safe to interpret.

The checks are deliberately read-only.  They return a structured report so
they can be used by CI, the command line, and the pipeline completion step.
"""

from __future__ import annotations

import json
import math
import os
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.contracts import normalize_od_demand
from src.network_artifact import load_network_artifact


class OutputValidationError(RuntimeError):
    """Raised by :func:`assert_experiment_outputs` for invalid artifacts."""


def _finite(value: Any) -> bool:
    try:
        return bool(math.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def _load_json(path: Path, errors: list[str]) -> dict[str, Any] | None:
    try:
        with path.open() as handle:
            value = json.load(handle)
        if not isinstance(value, dict):
            errors.append(f"{path}: expected a JSON object")
            return None
        return value
    except Exception as exc:  # pragma: no cover - message is the useful part
        errors.append(f"{path}: cannot read JSON ({exc})")
        return None


def _load_pickle(path: Path, errors: list[str]) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except Exception as exc:  # pragma: no cover - message is the useful part
        errors.append(f"{path}: cannot read pickle ({exc})")
        return None


def _od_tuple(value: Any) -> tuple[int, int] | None:
    if isinstance(value, str):
        try:
            left, right = value.replace("_", ",").split(",")
            return int(left), int(right)
        except (ValueError, TypeError):
            return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return int(value[0]), int(value[1])
        except (TypeError, ValueError):
            return None
    return None


def _route_records(config_data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return all route records found in one CG configuration."""
    reconstruction = config_data.get("reconstruction_results") or {}
    metrics = reconstruction.get("k_metrics") or {}
    records: list[Mapping[str, Any]] = []
    for metric in metrics.values():
        if isinstance(metric, Mapping):
            for route in metric.get("routes", []) or []:
                if isinstance(route, Mapping):
                    records.append(route)
    return records


def _validate_optimization_results(
    experiment_dir: Path,
    data: Any,
    network_hash: str,
    edge_ids: set[int],
    node_ids: set[int],
    errors: list[str],
) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        errors.append("all_optimization_results.pkl: expected a mapping")
        return {"configuration_count": 0, "route_count": 0}

    if data.get("network_hash") != network_hash:
        errors.append(
            "all_optimization_results.pkl: network_hash does not match canonical artifact"
        )
    run_configuration = data.get("run_configuration")
    if not isinstance(run_configuration, Mapping):
        errors.append("all_optimization_results.pkl: run_configuration is missing")
        run_configuration = {}
    elif run_configuration.get("network_hash") != network_hash:
        errors.append("all_optimization_results.pkl: run_configuration network_hash mismatch")
    solver = run_configuration.get("solver") if isinstance(run_configuration, Mapping) else None
    if isinstance(solver, Mapping) and solver.get("success") is False:
        errors.append("all_optimization_results.pkl: solver reported success=false")

    configurations = data.get("configurations")
    if not isinstance(configurations, Mapping) or not configurations:
        errors.append("all_optimization_results.pkl: no optimization configurations")
        configurations = {}

    connectivity = data.get("network_link_connectivity")
    allowed_edge_ids = set(edge_ids)
    if not isinstance(connectivity, list):
        errors.append("all_optimization_results.pkl: network_link_connectivity is missing")
    else:
        connectivity_ids = set()
        for entry in connectivity:
            if not isinstance(entry, Mapping) or "link_id" not in entry:
                errors.append("all_optimization_results.pkl: malformed link connectivity entry")
                continue
            link_id = int(entry["link_id"])
            connectivity_ids.add(link_id)
            if link_id not in edge_ids:
                # Charger self-links are derived by the CG stage and are not
                # part of the immutable road artifact.  They are valid only
                # when explicitly represented as a self-loop.
                if int(entry.get("start_node_id", -1)) != int(entry.get("end_node_id", -2)):
                    errors.append(
                        f"all_optimization_results.pkl: derived link {link_id} is not a self-link"
                    )
                else:
                    allowed_edge_ids.add(link_id)
        if not edge_ids.issubset(connectivity_ids):
            errors.append(
                "all_optimization_results.pkl: connectivity does not cover all "
                "canonical edge IDs"
            )

    route_count = 0
    finite_objectives = 0
    for key, config_data in configurations.items():
        if not isinstance(config_data, Mapping):
            errors.append(f"optimization configuration {key!r}: expected a mapping")
            continue

        # Road links are shared by every configuration, while charger
        # self-links are derived after a charger set is selected.  Validate
        # each configuration against its own complete connectivity table so
        # a link ID that is valid for one charger set cannot be mistaken for
        # the same edge in another set.
        config_connectivity = config_data.get("link_connectivity")
        if not isinstance(config_connectivity, list):
            # Backward-compatible import for older result pickles: the flow
            # records already carry the exact endpoint identity needed to
            # reconstruct this table.  New writers always persist it.
            raw_flows = config_data.get("link_flows")
            if isinstance(raw_flows, Mapping):
                config_connectivity = [
                    {
                        "link_id": int(raw_link_id),
                        "start_node_id": int(flow_data.get("start_node_id", -1)),
                        "end_node_id": int(flow_data.get("end_node_id", -2)),
                    }
                    for raw_link_id, flow_data in raw_flows.items()
                    if isinstance(flow_data, Mapping)
                ]
            else:
                config_connectivity = []

        config_allowed_edge_ids = set(edge_ids)
        config_connectivity_ids = set()
        for entry in config_connectivity:
            if not isinstance(entry, Mapping) or "link_id" not in entry:
                errors.append(f"optimization configuration {key!r}: malformed link connectivity entry")
                continue
            link_id = int(entry["link_id"])
            config_connectivity_ids.add(link_id)
            if link_id not in edge_ids:
                start_node = int(entry.get("start_node_id", -1))
                end_node = int(entry.get("end_node_id", -2))
                if start_node != end_node:
                    errors.append(
                        f"optimization configuration {key!r}: derived link {link_id} is not a self-link"
                    )
                else:
                    config_allowed_edge_ids.add(link_id)
        if not edge_ids.issubset(config_connectivity_ids):
            errors.append(
                f"optimization configuration {key!r}: connectivity does not cover all canonical edge IDs"
            )

        objective = config_data.get("objective_value", config_data.get("objective"))
        if not _finite(objective):
            errors.append(f"optimization configuration {key!r}: objective is not finite")
        else:
            finite_objectives += 1

        link_flows = config_data.get("link_flows")
        if not isinstance(link_flows, Mapping):
            errors.append(f"optimization configuration {key!r}: link_flows is missing")
        else:
            flow_ids = set()
            for raw_link_id, flow_data in link_flows.items():
                try:
                    link_id = int(raw_link_id)
                except (TypeError, ValueError):
                    errors.append(f"optimization configuration {key!r}: invalid link ID {raw_link_id!r}")
                    continue
                flow_ids.add(link_id)
                if link_id not in config_allowed_edge_ids:
                    errors.append(f"optimization configuration {key!r}: unknown link ID {link_id}")
                if not isinstance(flow_data, Mapping):
                    errors.append(f"optimization configuration {key!r}: malformed flow for link {link_id}")
                    continue
                if not _finite(flow_data.get("total_flow", 0.0)):
                    errors.append(f"optimization configuration {key!r}: non-finite flow on link {link_id}")
                elif float(flow_data.get("total_flow", 0.0)) < -1e-9:
                    errors.append(f"optimization configuration {key!r}: negative flow on link {link_id}")
            if flow_ids != config_allowed_edge_ids:
                errors.append(
                    f"optimization configuration {key!r}: link-flow IDs do not exactly cover "
                    "the canonical and derived optimization links"
                )

        for route in _route_records(config_data):
            route_count += 1
            route_id = route.get("route_id")
            if route_id is None:
                errors.append(f"optimization configuration {key!r}: route is missing route_id")
            if not _finite(route.get("flow", 0.0)) or float(route.get("flow", 0.0)) < -1e-9:
                errors.append(f"optimization configuration {key!r}: route has invalid flow")
            od = _od_tuple((route.get("origin"), route.get("destination")))
            if od is None or od[0] not in node_ids or od[1] not in node_ids:
                errors.append(f"optimization configuration {key!r}: route has invalid OD nodes")
            link_ids = route.get("link_ids")
            if not isinstance(link_ids, (list, tuple)) or not link_ids:
                errors.append(f"optimization configuration {key!r}: route {route_id!r} has no link_ids")
            else:
                unknown = []
                for raw_link_id in link_ids:
                    try:
                        link_id = int(raw_link_id)
                    except (TypeError, ValueError):
                        unknown.append(raw_link_id)
                        continue
                    if link_id not in config_allowed_edge_ids:
                        unknown.append(link_id)
                if unknown:
                    errors.append(
                        f"optimization configuration {key!r}: route {route_id!r} "
                        f"uses unknown link IDs {unknown[:5]}"
                    )

    if finite_objectives == 0:
        errors.append("all_optimization_results.pkl: no configuration has a finite objective")
    try:
        demand_class_count = (
            len(normalize_od_demand(run_configuration.get("od_demand", {})))
            if run_configuration.get("od_demand")
            else 0
        )
    except Exception as exc:
        errors.append(f"all_optimization_results.pkl: invalid demand contract ({exc})")
        demand_class_count = 0
    return {
        "configuration_count": len(configurations),
        "finite_objective_count": finite_objectives,
        "route_count": route_count,
        "demand_class_count": demand_class_count,
    }


def _assignment_count(value: Any) -> int:
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            return int(sum(int(item) for item in value))
        except (TypeError, ValueError):
            return 0
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _validate_queue_outputs(
    queue_dir: Path,
    network_hash: str,
    all_opt_data: Mapping[str, Any],
    errors: list[str],
) -> dict[str, Any]:
    required = ("queue_manifest.json", "NE_path_assignments.pkl", "comparison_results.json")
    for name in required:
        if not (queue_dir / name).exists():
            errors.append(f"queue/{name}: required queue artifact is missing")

    manifest = _load_json(queue_dir / "queue_manifest.json", errors) if (queue_dir / "queue_manifest.json").exists() else None
    if manifest is not None:
        if manifest.get("network_hash") != network_hash:
            errors.append("queue_manifest.json: network_hash mismatch")
        if manifest.get("failed_configurations"):
            errors.append("queue_manifest.json: failed_configurations is not empty")

    ne = _load_pickle(queue_dir / "NE_path_assignments.pkl", errors) if (queue_dir / "NE_path_assignments.pkl").exists() else None
    expected_classes: dict[tuple[tuple[int, int], str], int] = {}
    run_configuration = all_opt_data.get("run_configuration", {}) if isinstance(all_opt_data, Mapping) else {}
    try:
        for record in normalize_od_demand(run_configuration.get("od_demand", {})):
            expected_classes[((record.origin, record.destination), record.vehicle_type)] = record.demand
    except Exception as exc:
        errors.append(f"queue validation: invalid run_configuration demand ({exc})")

    nonconverged: list[str] = []
    approximate: list[str] = []
    if isinstance(ne, Mapping):
        if manifest and manifest.get("configuration_count") != len(ne):
            errors.append("queue_manifest.json: configuration_count does not match NE pickle")
        for placement, result in ne.items():
            if not isinstance(result, Mapping):
                errors.append(f"NE placement {placement!r}: expected a mapping")
                continue
            if result.get("network_hash") != network_hash:
                errors.append(f"NE placement {placement!r}: network_hash mismatch")
            if result.get("status") == "failed":
                errors.append(f"NE placement {placement!r}: status=failed")
            if result.get("status") == "approximate_cycle_state":
                approximate.append(str(placement))
            elif not bool(result.get("converged", False)):
                nonconverged.append(str(placement))
            assignments = result.get("assignments", {})
            if not isinstance(assignments, Mapping):
                errors.append(f"NE placement {placement!r}: assignments missing")
                continue
            for vehicle_type in ("F1", "F2"):
                type_assignments = assignments.get(vehicle_type, {})
                if not isinstance(type_assignments, Mapping):
                    errors.append(f"NE placement {placement!r}: {vehicle_type} assignments malformed")
                    continue
                observed: dict[tuple[int, int], int] = {}
                for raw_od, counts in type_assignments.items():
                    od = _od_tuple(raw_od)
                    if od is None:
                        errors.append(f"NE placement {placement!r}: invalid OD assignment key {raw_od!r}")
                        continue
                    observed[od] = observed.get(od, 0) + _assignment_count(counts)
                for (od, expected_type), expected in expected_classes.items():
                    if expected_type == vehicle_type and observed.get(od, 0) != expected:
                        errors.append(
                            f"NE placement {placement!r}: {vehicle_type} demand mismatch for {od}; "
                            f"expected {expected}, got {observed.get(od, 0)}"
                        )

    comparison = _load_json(queue_dir / "comparison_results.json", errors) if (queue_dir / "comparison_results.json").exists() else None
    if comparison is not None:
        if comparison.get("network_hash") != network_hash:
            errors.append("comparison_results.json: network_hash mismatch")
        for field in ("best_greedy", "best_exhaustive", "suboptimality_pct"):
            if field not in comparison:
                errors.append(f"comparison_results.json: missing {field}")
        for field in ("best_greedy", "best_exhaustive"):
            value = comparison.get(field, {})
            if isinstance(value, Mapping) and not _finite(value.get("avg_travel_time")):
                errors.append(f"comparison_results.json: {field}.avg_travel_time is not finite")

    return {
        "configuration_count": len(ne) if isinstance(ne, Mapping) else 0,
        "comparison_present": comparison is not None,
        "nonconverged_configurations": nonconverged,
        "approximate_configurations": approximate,
        "exact_ne_eligible": not approximate and not nonconverged,
    }


def validate_experiment_outputs(
    experiment_dir: str | os.PathLike[str],
    *,
    require_cg: bool = True,
    require_queue: bool = False,
    require_plots: bool = True,
    require_reports: bool = True,
) -> dict[str, Any]:
    """Validate one experiment directory and return a structured report.

    ``require_cg=False`` is useful for ``--network-only`` runs.  Queue checks
    are opt-in because queue simulation is an environment-dependent stage.
    The function never raises for a malformed result; callers can decide
    whether a failed validation should abort a run.  Use
    :func:`assert_experiment_outputs` when an exception is preferred.
    """
    root = Path(experiment_dir)
    errors: list[str] = []
    required = ["run_config.json", "network_manifest.json"]
    if require_reports:
        required.extend(["run_manifest.json", "experiment_summary.json", "report.md", "run_summary.txt"])
    if require_plots:
        required.append("plots/pruning_phases.png")
    if require_cg:
        required.append("all_optimization_results.pkl")
    for relative in required:
        if not (root / relative).exists():
            errors.append(f"{relative}: required artifact is missing")

    artifact_dir = root / "network"
    network_hash = None
    edge_ids: set[int] = set()
    node_ids: set[int] = set()
    network_counts = {}
    try:
        nodes, edges, manifest = load_network_artifact(artifact_dir)
        network_hash = manifest["network_hash"]
        edge_ids = {int(value) for value in edges["link_id"]}
        node_ids = {int(value) for value in nodes["node_id"]}
        network_counts = {"nodes": len(nodes), "edges": len(edges)}
        if edge_ids != set(range(len(edges))):
            errors.append("canonical network link IDs are not contiguous")
        if node_ids != set(range(len(nodes))):
            errors.append("canonical network node IDs are not contiguous")
        root_manifest = _load_json(root / "network_manifest.json", errors) if (root / "network_manifest.json").exists() else None
        if root_manifest is not None and root_manifest.get("network_hash") != network_hash:
            errors.append("root network_manifest.json: network_hash mismatch")
    except Exception as exc:
        errors.append(f"network artifact: {exc}")

    all_opt_data = None
    optimization_summary: dict[str, Any] = {}
    bpr_summary: dict[str, Any] = {}
    bpr_manifest_path = root / "bpr" / "bpr_manifest.json"
    if bpr_manifest_path.exists() and network_hash is not None:
        bpr_manifest = _load_json(bpr_manifest_path, errors)
        if bpr_manifest is not None:
            if bpr_manifest.get("network_hash") != network_hash:
                errors.append("bpr_manifest.json: network_hash mismatch")
            bpr_summary = {
                "successful_links": bpr_manifest.get("successful_links"),
                "failure_count": len(bpr_manifest.get("failures", [])),
                "fit_execution": bpr_manifest.get("fit_execution", {}),
            }
    bpr_csv = root / "bpr" / "traffic_data.csv"
    if bpr_csv.exists() and edge_ids:
        try:
            import pandas as pd

            bpr_ids = {int(value) for value in pd.read_csv(bpr_csv)["link_id"]}
            if bpr_ids != edge_ids:
                errors.append("bpr/traffic_data.csv: link IDs do not exactly cover canonical edges")
            bpr_summary["link_count"] = len(bpr_ids)
        except Exception as exc:
            errors.append(f"bpr/traffic_data.csv: cannot validate ({exc})")
    all_opt_path = root / "all_optimization_results.pkl"
    if require_cg and all_opt_path.exists() and network_hash is not None:
        all_opt_data = _load_pickle(all_opt_path, errors)
        optimization_summary = _validate_optimization_results(
            root, all_opt_data, network_hash, edge_ids, node_ids, errors
        )

    queue_summary: dict[str, Any] = {}
    if require_queue:
        if all_opt_data is None and all_opt_path.exists():
            all_opt_data = _load_pickle(all_opt_path, errors)
        if network_hash is not None and isinstance(all_opt_data, Mapping):
            queue_summary = _validate_queue_outputs(root / "queue", network_hash, all_opt_data, errors)
        else:
            errors.append("queue validation: cannot validate without a canonical network and CG results")

    run_manifest_path = root / "run_manifest.json"
    if run_manifest_path.exists() and network_hash is not None:
        run_manifest = _load_json(run_manifest_path, errors)
        provenance = run_manifest.get("provenance", {}) if isinstance(run_manifest, Mapping) else {}
        if provenance and provenance.get("network_hash") != network_hash:
            errors.append("run_manifest.json: provenance network_hash mismatch")

    return {
        "valid": not errors,
        "experiment_dir": str(root),
        "network_hash": network_hash,
        "network": network_counts,
        "bpr": bpr_summary,
        "optimization": optimization_summary,
        "queue": queue_summary,
        "errors": errors,
    }


def assert_experiment_outputs(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Validate outputs and raise a concise error if any invariant fails."""
    result = validate_experiment_outputs(*args, **kwargs)
    if not result["valid"]:
        details = "\n".join(f"- {error}" for error in result["errors"])
        raise OutputValidationError(f"Experiment output validation failed:\n{details}")
    return result
