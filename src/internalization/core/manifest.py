from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .types import digest


@dataclass(frozen=True)
class TaskManifest:
    benchmark: str
    environment_revision: str
    partitions: dict[str, tuple[str, ...]]

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text())
        obj = cls(data["benchmark"], data["environment_revision"],
                  {k: tuple(v) for k, v in data["partitions"].items()})
        obj.validate()
        if "manifest_hash" in data and data["manifest_hash"] != obj.fingerprint:
            raise ValueError("Manifest content/hash mismatch")
        return obj

    def validate(self):
        if not self.benchmark or not self.environment_revision:
            raise ValueError("Benchmark and pinned environment revision are required")
        used = set()
        for partition, ids in self.partitions.items():
            if not ids or any(not isinstance(i, str) or not i for i in ids):
                raise ValueError(f"Partition {partition} needs real task IDs")
            if len(ids) != len(set(ids)) or used.intersection(ids):
                raise ValueError("Duplicate task IDs or overlap between partitions")
            used.update(ids)

    @property
    def fingerprint(self):
        return digest({"benchmark": self.benchmark, "revision": self.environment_revision,
                       "partitions": self.partitions})

    def partition(self, name):
        if name.startswith("test"):
            raise ValueError("Outer-loop selection cannot access final test partitions")
        return self.partitions[name]

    def validate_loop(self, cycles, *, versioned=False, cohort_minimum=None):
        """Validate the entire planned allocation before any rollout or proposal."""
        self.validate()
        names = ("train", *loop_cohort_names(cycles, versioned=versioned))
        missing = [name for name in names if name not in self.partitions]
        if missing: raise ValueError("Incomplete loop manifest; missing partitions: " + ", ".join(missing))
        if cohort_minimum is not None:
            small = [name for name in names if name.startswith(("acceptance_", "retirement_"))
                     and len(self.partitions[name]) < cohort_minimum]
            if small: raise ValueError("Insufficient independent tasks in cohorts: " + ", ".join(small))


def loop_cohort_names(cycles, *, versioned=True):
    if type(cycles) is not int or cycles < 1: raise ValueError("cycles must be a positive integer")
    return ("search", "dev", *(f"retirement_{i}" for i in range(cycles)),
            *((f"acceptance_{i}" for i in range(cycles)) if versioned else ()))
