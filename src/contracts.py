"""Shared, dependency-light contracts for reproducible pipeline runs.

The project historically passed loosely shaped dictionaries between the
congestion-game and queue stages.  This module keeps the public configuration
backwards compatible while giving every stage the same normalized demand,
route, seed, and timing primitives.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


VEHICLE_TYPES = ("F1", "F2")


@dataclass(frozen=True)
class DemandClass:
    """One OD/type demand class.

    F1 is a never-charge vehicle and F2 is a vehicle that must charge once.
    The type field is intentionally a string so F3 can be added without
    changing the serialized contract.
    """

    od_id: str
    origin: int
    destination: int
    vehicle_type: str
    demand: int


def _parse_od_key(key: Any) -> tuple[int, int]:
    if isinstance(key, str):
        parts = [part.strip() for part in key.split(",")]
    elif isinstance(key, Sequence) and len(key) == 2:
        parts = list(key)
    else:
        raise ValueError(f"OD key must be 'origin,destination' or a pair: {key!r}")
    try:
        origin, destination = int(parts[0]), int(parts[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Invalid OD key: {key!r}") from exc
    if origin == destination:
        raise ValueError(f"OD origin and destination must differ: {key!r}")
    return origin, destination


def normalize_od_demand(raw: Any) -> list[DemandClass]:
    """Normalize legacy or typed demand into deterministic F1/F2 records.

    Supported inputs:

    * ``{"7,26": [60, 120]}`` (legacy F1/F2 shorthand)
    * ``{(7, 26): {"F1": 60, "F2": 120}}``
    * a list of records containing origin, destination, type, and demand
    * a list of records containing ``classes``
    """
    records: list[DemandClass] = []

    def add(origin: int, destination: int, vehicle_type: str, demand: Any) -> None:
        vehicle_type = str(vehicle_type).upper()
        if vehicle_type not in VEHICLE_TYPES:
            raise ValueError(
                f"Unsupported vehicle type {vehicle_type!r}; supported types are {VEHICLE_TYPES}"
            )
        if isinstance(demand, bool):
            raise ValueError("Demand must be a non-negative integer")
        try:
            numeric = float(demand)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid demand {demand!r}") from exc
        if numeric < 0 or not numeric.is_integer():
            raise ValueError(f"Demand must be a non-negative integer: {demand!r}")
        records.append(
            DemandClass(
                od_id=f"{origin},{destination}",
                origin=int(origin),
                destination=int(destination),
                vehicle_type=vehicle_type,
                demand=int(numeric),
            )
        )

    if isinstance(raw, Mapping):
        for key, value in raw.items():
            origin, destination = _parse_od_key(key)
            if isinstance(value, Mapping):
                for vehicle_type, demand in value.items():
                    add(origin, destination, vehicle_type, demand)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                if len(value) != 2:
                    raise ValueError(f"od_demand['{key}'] must be [F1, F2]")
                add(origin, destination, "F1", value[0])
                add(origin, destination, "F2", value[1])
            else:
                raise ValueError(f"Invalid demand value for {key!r}: {value!r}")
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError(f"Demand records must be mappings: {item!r}")
            origin, destination = _parse_od_key(
                item.get("od_id", (item.get("origin"), item.get("destination")))
            )
            if "classes" in item:
                classes = item["classes"]
                if not isinstance(classes, Mapping):
                    raise ValueError("Demand 'classes' must be a mapping")
                for vehicle_type, demand in classes.items():
                    add(origin, destination, vehicle_type, demand)
            else:
                add(
                    origin,
                    destination,
                    item.get("type", item.get("vehicle_type")),
                    item.get("demand"),
                )
    else:
        raise ValueError("OD demand must be a mapping or a list of records")

    records.sort(key=lambda r: (r.origin, r.destination, r.vehicle_type))
    if not records:
        raise ValueError("At least one OD demand class is required")
    return records


def demand_by_od(records: Iterable[DemandClass]) -> dict[tuple[int, int], dict[str, int]]:
    result: dict[tuple[int, int], dict[str, int]] = {}
    for record in records:
        result.setdefault((record.origin, record.destination), {})[record.vehicle_type] = record.demand
    return result


def stable_json(value: Any) -> str:
    """Serialize values deterministically for hashes and provenance."""

    def default(obj: Any) -> Any:
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        return str(obj)

    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=default)


class SeedManager:
    """Deterministic named random streams shared by all pipeline stages."""

    def __init__(self, seed: int | None):
        self.seed = int(seed if seed is not None else 0)
        self.apply_global()

    def apply_global(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)

    def derive(self, namespace: str, *parts: Any) -> int:
        payload = stable_json([self.seed, namespace, *parts]).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32 - 1)

    def numpy(self, namespace: str, *parts: Any) -> np.random.Generator:
        return np.random.default_rng(self.derive(namespace, *parts))


class TimingRecorder:
    """Collect named wall-clock timings without coupling stages together."""

    def __init__(self) -> None:
        self.values: dict[str, float] = {}
        self.events: list[dict[str, Any]] = []

    def measure(self, name: str):
        return _TimingContext(self, name)

    def add(self, name: str, elapsed: float, **metadata: Any) -> None:
        self.values[name] = self.values.get(name, 0.0) + float(elapsed)
        event = {"name": name, "seconds": float(elapsed)}
        event.update(metadata)
        self.events.append(event)


class _TimingContext:
    def __init__(self, recorder: TimingRecorder, name: str):
        self.recorder = recorder
        self.name = name
        self.started = 0.0

    def __enter__(self):
        self.started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.recorder.add(self.name, time.perf_counter() - self.started, error=exc is not None)
        return False
