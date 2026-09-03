"""Centralized configuration management for the EV charger optimization pipeline."""
import json
from dataclasses import dataclass, field
from typing import Optional

from src.contracts import DemandClass, normalize_od_demand


QUEUE_DEFAULTS = {
    "enabled": True,
    "K": 16,
    "THRESH": 100,
    "ALPHA": 0.01,
    "NUM_ITERS": 50,
    "ENT_CAPACITY": 250,
    "CHARGING_CAPACITY": 250,
    "EXIT_CAPACITY": 250,
    "COST": 0,
    "SIMULATION_HORIZON": 10801,
    "MAX_NE_ITERATIONS": 200,
    "WORKERS": None,
    "N": 750,
    "single_swap": True,
    "failure_policy": "fail_fast",
}

PIPELINE_DEFAULTS = {
    "random_seed": 42,
    # None means use os.cpu_count() for stages whose local worker count is
    # also unset.  Individual stages may override this with an integer.
    "parallel_workers": None,
    "skip_bpr_fitting": False,
    "skip_cg_optimization": False,
    "skip_queue_simulation": False,
    # CG remains end-to-end by default, but records any degraded BPR
    # provenance.  Strict policies are opt-in for final/paper runs.
    "cg_fit_policy": "allow_degraded",
    "bpr_generation": {
        # The historical artifact is the regression reference for the BPR
        # stage.  The strict capacity-fraction implementation remains
        # available by setting mode=capacity_fraction_strict.
        "mode": "historical_artifact_compatible",
        "num_samples": 25,
        "max_flow": 250,
        "flow_fractions": None,
        "fit_validation": "parameter_complete",
        "fallback_policy": "legacy_proxy_and_constant",
        "historical_reference_commit": "37eab33",
        "capacity_source": "simulator",
        "capacity_per_lane": 1900.0,
        "calibration_window_hours": 0.1,
        "route_mode": "contextual",
        "fixed_references": False,
        "workers": None,
        "fit_workers": None,
        "save_fit_plots": True,
        "timeout": None,
        "failure_policy": "proxy",
        "allow_proxy": True,
        "missing_context_policy": "synthetic_boundary",
        "synthetic_context_capacity_multiplier": 10.0,
        "synthetic_context_length_m": 1.0,
        "simulation_horizon": 10801,
        "require_full_fit": False,
        "min_r2": 0.0,
        # The pipeline default attempts a nonlinear fit for every finite
        # observation vector.  Set fit_screening=legacy and
        # accept_low_r2=false for exact historical rejection behavior.
        "fit_screening": "none",
        "correlation_threshold": 0.0,
        "variation_ratio_threshold": 0.0,
        "accept_low_r2": True,
        "force_regenerate": False,
        "min_samples": 2,
    },
    "artifact_dir": None,
}

ROAD_FILTER_DEFAULTS = {
    "enabled": True,
    "highway_types": [
        "motorway", "trunk", "primary", "secondary",
        "motorway_link", "trunk_link", "primary_link", "secondary_link",
    ],
    "prune_dead_ends": False,
    "merge_chains": True,
    "suppress_t_junctions": False,
    "contract_threshold": 30,
    "cross_threshold": 200,
}


@dataclass
class Config:
    coordinates: list
    num_chargers: int
    possible_charger_positions: list
    od_demand: dict
    max_iter: int = 1000
    use_derivatives: bool = False
    single_swap: bool = True
    use_cvxpy: bool = True
    plot_info: bool = False
    calculate_on_all_possible_positions: bool = True
    charger_self_link_length: float = 100.0
    route_analysis: dict = field(default_factory=dict)
    queue_simulation: dict = field(default_factory=lambda: dict(QUEUE_DEFAULTS))
    pipeline: dict = field(default_factory=lambda: dict(PIPELINE_DEFAULTS))
    road_filter: dict = field(default_factory=lambda: dict(ROAD_FILTER_DEFAULTS))

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, "r") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        queue = {**QUEUE_DEFAULTS, **raw.get("queue_simulation", {})}
        pipeline = {**PIPELINE_DEFAULTS, **raw.get("pipeline", {})}
        raw_bpr = raw.get("pipeline", {}).get("bpr_generation", {})
        # Configurations written before the versioned mode existed are
        # unambiguous when they explicitly request fixed references or flow
        # fractions.  Preserve those runs as strict mode; a configuration
        # without those legacy keys receives the new historical default.
        inferred_bpr_mode = None
        if "mode" not in raw_bpr and (
            "fixed_references" in raw_bpr or "flow_fractions" in raw_bpr
        ):
            inferred_bpr_mode = "capacity_fraction_strict"
        strict_bpr_defaults = {}
        if raw_bpr.get("mode", inferred_bpr_mode) == "capacity_fraction_strict":
            strict_bpr_defaults = {
                "mode": "capacity_fraction_strict",
                "flow_fractions": [0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 4.0, 10.0],
                "fit_validation": "full",
                "fallback_policy": "none",
                "route_mode": "link_probe",
                "fixed_references": True,
                "failure_policy": "fail_fast",
                "allow_proxy": False,
                "require_full_fit": True,
            }
        pipeline["bpr_generation"] = {
            **PIPELINE_DEFAULTS["bpr_generation"],
            **strict_bpr_defaults,
            **raw_bpr,
        }
        road_filter = {**ROAD_FILTER_DEFAULTS, **raw.get("road_filter", {})}
        cfg = cls(
            coordinates=raw["coordinates"],
            num_chargers=raw["num_chargers"],
            possible_charger_positions=[int(value) for value in raw["possible_charger_positions"]],
            od_demand=raw["od_demand"],
            max_iter=raw.get("max_iter", 1000),
            use_derivatives=raw.get("use_derivatives", False),
            single_swap=raw.get("single_swap", True),
            use_cvxpy=raw.get("use_cvxpy", True),
            plot_info=raw.get("plot_info", False),
            calculate_on_all_possible_positions=raw.get("calculate_on_all_possible_positions", True),
            charger_self_link_length=float(raw.get("charger_self_link_length", 100.0)),
            route_analysis=raw.get("route_analysis", {}),
            queue_simulation=queue,
            pipeline=pipeline,
            road_filter=road_filter,
        )
        cfg._explicit_queue_config = "queue_simulation" in raw
        cfg._validate()
        return cfg

    def _validate(self):
        if len(self.coordinates) != 4:
            raise ValueError("coordinates must be [north, south, east, west]")
        if self.num_chargers < 1:
            raise ValueError("num_chargers must be >= 1")
        if not self.possible_charger_positions:
            raise ValueError("possible_charger_positions must not be empty")
        if self.num_chargers > len(self.possible_charger_positions):
            raise ValueError("num_chargers cannot exceed len(possible_charger_positions)")
        if self.charger_self_link_length < 0:
            raise ValueError("charger_self_link_length must be non-negative")
        records = normalize_od_demand(self.od_demand)
        valid_nodes = set(self.possible_charger_positions)
        if any(record.demand < 0 for record in records):
            raise ValueError("OD demand must be non-negative")
        q = self.queue_simulation
        if q["K"] < 1:
            raise ValueError("queue_simulation.K must be >= 1")
        if q["THRESH"] < 0:
            raise ValueError("queue_simulation.THRESH must be >= 0")
        if q["NUM_ITERS"] < 1:
            raise ValueError("queue_simulation.NUM_ITERS must be >= 1")
        if q["N"] < 1:
            raise ValueError("queue_simulation.N must be >= 1")
        if int(q.get("SIMULATION_HORIZON", 10801)) < 1:
            raise ValueError("queue_simulation.SIMULATION_HORIZON must be >= 1")
        if int(q.get("MAX_NE_ITERATIONS", 200)) < 1:
            raise ValueError("queue_simulation.MAX_NE_ITERATIONS must be >= 1")
        if not 0 <= float(q["ALPHA"]):
            raise ValueError("queue_simulation.ALPHA must be >= 0")
        route_k_values = self.route_analysis.get("k_values", [])
        if (
            getattr(self, "_explicit_queue_config", False)
            and route_k_values
            and int(q["K"]) not in {int(value) for value in route_k_values}
        ):
            raise ValueError(
                "queue_simulation.K must be included in route_analysis.k_values"
            )
        if q.get("failure_policy") not in {"fail_fast", "record", "inf"}:
            raise ValueError("queue_simulation.failure_policy must be fail_fast, record, or inf")
        bpr = self.pipeline.get("bpr_generation", {})
        if bpr.get("mode") not in {"historical_artifact_compatible", "capacity_fraction_strict"}:
            raise ValueError(
                "pipeline.bpr_generation.mode must be historical_artifact_compatible "
                "or capacity_fraction_strict"
            )
        if bpr.get("fit_validation", "full") not in {"full", "parameter_complete"}:
            raise ValueError(
                "pipeline.bpr_generation.fit_validation must be full or parameter_complete"
            )
        if bpr.get("fallback_policy", "none") not in {
            "none", "legacy_proxy_and_constant"
        }:
            raise ValueError(
                "pipeline.bpr_generation.fallback_policy must be none or "
                "legacy_proxy_and_constant"
            )
        if int(bpr.get("num_samples", 25)) < 2:
            raise ValueError("pipeline.bpr_generation.num_samples must be >= 2")
        if float(bpr.get("max_flow", 250)) <= 0:
            raise ValueError("pipeline.bpr_generation.max_flow must be positive")
        if bpr.get("failure_policy") not in {"fail_fast", "record", "proxy"}:
            raise ValueError("pipeline.bpr_generation.failure_policy must be fail_fast, record, or proxy")
        if bpr.get("missing_context_policy", "synthetic_boundary") not in {
            "synthetic_boundary", "proxy", "fail_fast"
        }:
            raise ValueError(
                "pipeline.bpr_generation.missing_context_policy must be "
                "synthetic_boundary, proxy, or fail_fast"
            )
        if float(bpr.get("synthetic_context_capacity_multiplier", 10.0)) <= 0:
            raise ValueError(
                "pipeline.bpr_generation.synthetic_context_capacity_multiplier must be positive"
            )
        if float(bpr.get("synthetic_context_length_m", 1.0)) <= 0:
            raise ValueError(
                "pipeline.bpr_generation.synthetic_context_length_m must be positive"
            )
        if int(bpr.get("simulation_horizon", 10801)) < 1:
            raise ValueError("pipeline.bpr_generation.simulation_horizon must be >= 1")
        if int(bpr.get("min_samples", 2)) < 2:
            raise ValueError("pipeline.bpr_generation.min_samples must be >= 2")
        if bpr.get("timeout") is not None and float(bpr["timeout"]) <= 0:
            raise ValueError("pipeline.bpr_generation.timeout must be positive")
        fractions = bpr.get("flow_fractions")
        if fractions is not None:
            fractions = [float(value) for value in fractions]
            if not fractions or any(value < 0 for value in fractions) or len(set(fractions)) != len(fractions):
                raise ValueError(
                    "pipeline.bpr_generation.flow_fractions must be unique non-negative values"
                )
            if bpr.get("capacity_source", "simulator") not in {"simulator", "artifact"}:
                raise ValueError(
                    "pipeline.bpr_generation.capacity_source must be simulator or artifact"
                )
            if float(bpr.get("capacity_per_lane", 1900.0)) <= 0:
                raise ValueError("pipeline.bpr_generation.capacity_per_lane must be positive")
            if float(bpr.get("calibration_window_hours", 1.0)) <= 0:
                raise ValueError("pipeline.bpr_generation.calibration_window_hours must be positive")
        if float(bpr.get("min_r2", 0.5)) < 0:
            raise ValueError("pipeline.bpr_generation.min_r2 must be non-negative")
        if bpr.get("fit_screening", "legacy") not in {"legacy", "none"}:
            raise ValueError(
                "pipeline.bpr_generation.fit_screening must be legacy or none"
            )
        if float(bpr.get("correlation_threshold", 0.3)) < -1 or float(
            bpr.get("correlation_threshold", 0.3)
        ) > 1:
            raise ValueError(
                "pipeline.bpr_generation.correlation_threshold must be in [-1, 1]"
            )
        if float(bpr.get("variation_ratio_threshold", 0.03)) < 0:
            raise ValueError(
                "pipeline.bpr_generation.variation_ratio_threshold must be non-negative"
            )
        if bpr.get("route_mode", "link_probe") not in {"link_probe", "isolated_link", "contextual"}:
            raise ValueError(
                "pipeline.bpr_generation.route_mode must be link_probe, isolated_link, or contextual"
            )
        if bpr.get("mode") == "historical_artifact_compatible":
            if bpr.get("fit_validation") != "parameter_complete":
                raise ValueError(
                    "historical_artifact_compatible mode requires "
                    "fit_validation=parameter_complete"
                )
            if bpr.get("fallback_policy") != "legacy_proxy_and_constant":
                raise ValueError(
                    "historical_artifact_compatible mode requires "
                    "fallback_policy=legacy_proxy_and_constant"
                )
            if bpr.get("fixed_references", False):
                raise ValueError(
                    "historical_artifact_compatible mode requires fixed_references=false"
                )
        elif bpr.get("mode") == "capacity_fraction_strict":
            if bpr.get("fit_validation") not in {"full", None}:
                raise ValueError(
                    "capacity_fraction_strict mode requires fit_validation=full"
                )
        if bpr.get("fit_workers") is not None and int(bpr["fit_workers"]) < 1:
            raise ValueError("pipeline.bpr_generation.fit_workers must be >= 1")
        if self.pipeline.get("parallel_workers") is not None and int(self.pipeline["parallel_workers"]) < 1:
            raise ValueError("pipeline.parallel_workers must be >= 1 when set")
        if self.pipeline.get("cg_fit_policy", "allow_degraded") not in {
            "allow_degraded", "reject_proxy_or_constant", "validated_only"
        }:
            raise ValueError(
                "pipeline.cg_fit_policy must be allow_degraded, "
                "reject_proxy_or_constant, or validated_only"
            )

    def to_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    def to_dict(self) -> dict:
        return {
            "coordinates": self.coordinates,
            "num_chargers": self.num_chargers,
            "possible_charger_positions": self.possible_charger_positions,
            "od_demand": self.od_demand,
            "max_iter": self.max_iter,
            "use_derivatives": self.use_derivatives,
            "single_swap": self.single_swap,
            "use_cvxpy": self.use_cvxpy,
            "plot_info": self.plot_info,
            "calculate_on_all_possible_positions": self.calculate_on_all_possible_positions,
            "charger_self_link_length": self.charger_self_link_length,
            "route_analysis": self.route_analysis,
            "queue_simulation": self.queue_simulation,
            "pipeline": self.pipeline,
            "road_filter": self.road_filter,
        }

    def get_od_demand_tuples(self) -> dict:
        result = {}
        for record in self.get_demand_classes():
            result.setdefault((record.origin, record.destination), {})[record.vehicle_type] = record.demand
        return {
            od: (classes.get("F1", 0), classes.get("F2", 0))
            for od, classes in result.items()
        }

    def get_demand_classes(self) -> list[DemandClass]:
        return normalize_od_demand(self.od_demand)

    def get_queue_param(self, key: str, default=None):
        return self.queue_simulation.get(key, default)

    def get_road_filter_param(self, key: str, default=None):
        return self.road_filter.get(key, default)
