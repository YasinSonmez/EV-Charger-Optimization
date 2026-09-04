"""Small atomic/provenance helpers shared by local and cluster runners."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import resource
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.contracts import stable_json


def config_digest(config: dict) -> str:
    return hashlib.sha256(stable_json(config).encode("utf-8")).hexdigest()


def safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-") or "experiment"


def atomic_write_json(path, value) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def available_cpus() -> int:
    for key in ("SLURM_CPUS_PER_TASK", "PBS_NP"):
        value = os.environ.get(key)
        if value:
            try:
                return max(1, int(value))
            except ValueError:
                pass
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def process_provenance() -> dict:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "available_cpus": available_cpus(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "containerized": os.environ.get("EVOPT_CONTAINERIZED") == "1",
        "container_image": os.environ.get("EVOPT_IMAGE_REF"),
        "container_image_digest": os.environ.get("EVOPT_IMAGE_DIGEST"),
        "execution_mode": os.environ.get("EVOPT_EXECUTION_MODE", "native"),
        "code_commit": os.environ.get("EVOPT_CODE_COMMIT"),
        "peak_rss_raw": usage.ru_maxrss,
    }


def directory_inventory(root) -> dict:
    base = Path(root)
    files = []
    total = 0
    if base.exists():
        for path in sorted(item for item in base.rglob("*") if item.is_file()):
            size = path.stat().st_size
            total += size
            files.append({"path": str(path.relative_to(base)), "bytes": size})
    return {"file_count": len(files), "total_bytes": total, "files": files}
