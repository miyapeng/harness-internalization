"""Compatibility export for standalone ALFWorld evaluation."""
from dataclasses import asdict
import json
from pathlib import Path
from .benchmarks.alfworld import AlfworldEnvironment
from .training.rollout import InteractionTaskRunner
from .training.teacher_backend import FrozenHFBackend
from .core.serialization import harness_from_dict
from .core.types import Cost, write_json

def evaluate_request(request_path, response_path, *, env_config=Path("configs/alfworld.yaml"), device="cpu", max_steps=30):
    request = json.loads(request_path.read_text())
    runner = InteractionTaskRunner(lambda: AlfworldEnvironment(env_config),
        lambda checkpoint: FrozenHFBackend(checkpoint, device=device), max_steps=max_steps)
    result = runner.rollout(request["checkpoint"], harness_from_dict(request["harness"]), tuple(request["task_ids"]),
                           seeds=tuple(request["seeds"]), output=response_path.parent)
    write_json(response_path, {"schema_version": 1, "episodes": [asdict(r) for r in result.evaluations],
        "cost": asdict(sum((t.cost for t in result.trajectories), Cost()))})
