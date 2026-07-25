"""Centralized configuration management for the EV charger optimization pipeline."""
import json
from dataclasses import dataclass, field
from typing import Optional


QUEUE_DEFAULTS = {
    "enabled": True,
    "K": 16,
    "THRESH": 100,
    "NUM_ITERS": 50,
    "ENT_CAPACITY": 250,
    "CHARGING_CAPACITY": 250,
    "EXIT_CAPACITY": 250,
    "COST": 0,
    "WORKERS": None,
    "N": 750,
    "single_swap": True,
}

PIPELINE_DEFAULTS = {
    "random_seed": 42,
    "skip_bpr_fitting": False,
    "skip_cg_optimization": False,
    "skip_queue_simulation": False,
    "bpr_generation": {
        "num_samples": 25,
        "max_flow": 250,
    },
}

ROAD_FILTER_DEFAULTS = {
    "enabled": True,
    "highway_types": [
        "motorway", "trunk", "primary", "secondary",
        "motorway_link", "trunk_link", "primary_link", "secondary_link",
    ],
    "prune_dead_ends": True,
    "merge_chains": True,
    "suppress_t_junctions": True,
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
        road_filter = {**ROAD_FILTER_DEFAULTS, **raw.get("road_filter", {})}
        cfg = cls(
            coordinates=raw["coordinates"],
            num_chargers=raw["num_chargers"],
            possible_charger_positions=raw["possible_charger_positions"],
            od_demand=raw["od_demand"],
            max_iter=raw.get("max_iter", 1000),
            use_derivatives=raw.get("use_derivatives", False),
            single_swap=raw.get("single_swap", True),
            use_cvxpy=raw.get("use_cvxpy", True),
            plot_info=raw.get("plot_info", False),
            calculate_on_all_possible_positions=raw.get("calculate_on_all_possible_positions", True),
            route_analysis=raw.get("route_analysis", {}),
            queue_simulation=queue,
            pipeline=pipeline,
            road_filter=road_filter,
        )
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
        for key, val in self.od_demand.items():
            if not isinstance(val, list) or len(val) != 2:
                raise ValueError(f"od_demand['{key}'] must be [non_charging, charging]")
        q = self.queue_simulation
        if q["K"] < 1:
            raise ValueError("queue_simulation.K must be >= 1")
        if q["THRESH"] < 0:
            raise ValueError("queue_simulation.THRESH must be >= 0")
        if q["NUM_ITERS"] < 1:
            raise ValueError("queue_simulation.NUM_ITERS must be >= 1")
        if q["N"] < 1:
            raise ValueError("queue_simulation.N must be >= 1")

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
            "route_analysis": self.route_analysis,
            "queue_simulation": self.queue_simulation,
            "pipeline": self.pipeline,
            "road_filter": self.road_filter,
        }

    def get_od_demand_tuples(self) -> dict:
        return {tuple(map(int, k.split(","))): tuple(v) for k, v in self.od_demand.items()}

    def get_queue_param(self, key: str, default=None):
        return self.queue_simulation.get(key, default)

    def get_road_filter_param(self, key: str, default=None):
        return self.road_filter.get(key, default)
