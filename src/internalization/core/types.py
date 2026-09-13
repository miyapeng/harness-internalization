from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Artifacts are immutable: never silently replace a previous experiment.
    with path.open("x") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


class Journal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path

    def append(self, kind: str, **payload):
        with self.path.open("a") as f:
            f.write(json.dumps({"kind": kind, **payload}, ensure_ascii=False, allow_nan=False) + "\n")


@dataclass(frozen=True)
class State:
    task_id: str
    episode_id: str
    step: int
    public_history: str
    # This must be exactly the student's visible prompt, not an untruncated archive.
    def __post_init__(self):
        if not self.task_id or not self.episode_id or self.step < 0:
            raise ValueError("Real task ID, episode ID and nonnegative step are required")

    @property
    def fingerprint(self):
        return digest(asdict(self))


@dataclass(frozen=True)
class Cost:
    input_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    auxiliary_calls: int = 0
    tool_calls: int = 0
    latency_s: float = 0.0

    def __post_init__(self):
        if any(not math.isfinite(v) or v < 0 for v in asdict(self).values()):
            raise ValueError("Costs must be finite and nonnegative")

    def __add__(self, other):
        return Cost(**{k: v + asdict(other)[k] for k, v in asdict(self).items()})

    @property
    def total_tokens(self):
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class EpisodeResult:
    task_id: str
    seed: int
    success: float
    cost: Cost

    def __post_init__(self):
        if not self.task_id or not math.isfinite(self.success) or not 0 <= self.success <= 1:
            raise ValueError("Invalid episode result")

    @property
    def key(self):
        return self.task_id, self.seed


def read_results(path: Path):
    return [EpisodeResult(**{**row, "cost": Cost(**row["cost"])}) for row in json.loads(path.read_text())]

# Canonical evaluation name; EpisodeResult remains a compatibility alias.
EvaluationResult = EpisodeResult
