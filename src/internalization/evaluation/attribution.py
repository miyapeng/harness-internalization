"""Pre-training contribution gate; never inferred from post-training scores."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from .retirement import paired_interval


@dataclass(frozen=True)
class AttributionPolicy:
    min_external_gain: float = 0.0
    confidence: float = 0.95
    bootstrap_samples: int = 2000
    min_tasks: int = 30
    seed: int = 42

    def __post_init__(self):
        if (not math.isfinite(self.min_external_gain) or not 0 <= self.min_external_gain < 1
            or not 0 < self.confidence < 1
            or type(self.bootstrap_samples) is not int or self.bootstrap_samples < 100
            or type(self.min_tasks) is not int or self.min_tasks < 2
            or type(self.seed) is not int):
            raise ValueError("Invalid pre-training attribution policy")

    @classmethod
    def load(cls, path: Path):
        return cls(**json.loads(path.read_text()))


def evaluate_attribution(a, b, policy=AttributionPolicy()):
    """Require enough paired tasks and a strictly positive lower gain bound."""
    interval = paired_interval(a, b, lambda row: row.success, policy)
    checks = {
        "enough_independent_tasks": interval["tasks"] >= policy.min_tasks,
        "external_contribution": interval["low"] > policy.min_external_gain,
    }
    passed = all(checks.values())
    return {"passed": passed, "decision": "internalize" if passed else "discard",
            "reason": None if passed else "no_external_contribution",
            "delta_external": interval, "checks": checks, "policy": asdict(policy),
            "success": {"A": mean(row.success for row in a), "B": mean(row.success for row in b)}}
