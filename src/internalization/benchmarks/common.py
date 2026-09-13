"""Local benchmark contracts; records and graders never become model context."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

BENCHMARKS = ("terminalbench2", "swebench_pro", "hotpotqa", "lawbench")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_hash(path):
    root = Path(path).resolve(strict=True)
    entries = []
    for file in sorted(root.rglob("*")):
        if file.is_symlink(): raise ValueError("Task bundles must not contain symlinks")
        if file.is_file(): entries.append((str(file.relative_to(root)), sha256(file)))
    return hashlib.sha256(json.dumps(entries).encode()).hexdigest()


def checked_repo(root, revision):
    root = Path(root).resolve(strict=True)
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError("An exact external evaluator git commit is required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, text=True)
    if head != revision or dirty: raise ValueError("External evaluator revision/working tree mismatch")
    return root


def score01(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Benchmark score must be finite and in [0, 1]")
    return value


@dataclass(frozen=True)
class BenchmarkConfig:
    benchmark: str
    catalog: str
    max_steps: int = 30
    timeout_s: int = 7200
    command_timeout_s: int = 120
    max_context: int = 32768
    max_prompt_tokens: int = 28672
    max_action_tokens: int = 2048
    max_new_tokens: int = 1024
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.benchmark not in BENCHMARKS: raise ValueError("Unknown benchmark")
        for key in ("max_steps", "timeout_s", "command_timeout_s", "max_context",
                    "max_prompt_tokens", "max_action_tokens", "max_new_tokens"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"Positive integer required: {key}")
        if self.max_prompt_tokens + max(self.max_action_tokens, self.max_new_tokens) > self.max_context:
            raise ValueError("Context budget exceeded")
        if self.timeout_s <= self.command_timeout_s: raise ValueError("RPC timeout must exceed command timeout")
        if self.benchmark == "lawbench" and self.max_steps != 1:
            raise ValueError("LawBench uses its native single-response protocol")

    @classmethod
    def load(cls, path):
        raw = os.path.expandvars(Path(path).read_text())
        if "${" in raw: raise ValueError("Unset benchmark configuration environment variable")
        return cls(**json.loads(raw))

    @property
    def model_options(self):
        return {k: getattr(self, k) for k in ("max_context", "max_action_tokens", "max_new_tokens")}


class Catalog:
    def __init__(self, path, benchmark, expected_hash=None):
        self.path = Path(path).resolve(strict=True)
        self.fingerprint = sha256(self.path)
        if expected_hash is not None and self.fingerprint != expected_hash:
            raise ValueError("Catalog changed during the stage")
        data = json.loads(self.path.read_text())
        if data["benchmark"] != benchmark or not data["revision"]:
            raise ValueError("Catalog benchmark/revision mismatch")
        self.revision = data["revision"]
        self.tasks = {r["id"]: r for r in data["tasks"]}
        if not self.tasks or len(self.tasks) != len(data["tasks"]):
            raise ValueError("Catalog must contain unique nonempty task IDs")
        for key, row in self.tasks.items():
            if not isinstance(key, str) or not key or row["split"] not in ("train", "dev", "test"):
                raise ValueError("Invalid task identity/split")

    def get(self, task_id, training=False):
        row = self.tasks[task_id]
        if training and row["split"] != "train": raise ValueError("Training requires a training task")
        return row

    def check_selection(self, tasks, *, final=False):
        for task in tasks:
            row = self.get(task)
            if row["split"] == "test" and not final:
                raise ValueError("Published held-out task cannot enter outer-loop search/retirement")


def action_json(text):
    value = json.loads(text)
    if not isinstance(value, dict) or not isinstance(value.get("action"), str):
        raise ValueError("Expected a JSON object with an action field")
    return value
