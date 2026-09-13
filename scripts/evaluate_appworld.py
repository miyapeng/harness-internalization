#!/usr/bin/env python3
"""Final AppWorld evaluation outside outer-loop selection; print aggregate metrics only."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from internalization.benchmarks.appworld_manifest import aggregate_results
from internalization.command_backend import CommandBackend
from internalization.core.manifest import TaskManifest
from internalization.core.serialization import harness_from_dict
from internalization.core.types import Cost, write_json
from internalization.harness.module import Harness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", type=Path, default=Path("configs/appworld_backend.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, help="Accepted cycle state.json, including residual Harness")
    parser.add_argument("--partition", choices=("dev", "test_normal", "test_challenge"), required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = TaskManifest.load(args.manifest)
    if manifest.benchmark != "appworld": raise ValueError("Expected an AppWorld manifest")
    if not args.seeds or len(args.seeds) != len(set(args.seeds)): raise ValueError("Unique seeds required")
    checkpoint = str(args.checkpoint.resolve(strict=True))
    harness = Harness()
    if args.state:
        state = json.loads(args.state.read_text())
        if str(Path(state["checkpoint"]).resolve(strict=True)) != checkpoint:
            raise ValueError("State must belong to the checkpoint being evaluated")
        harness = harness_from_dict({"version":state["harness_version"], "modules":state["active_modules"]})
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", {"manifest_hash":manifest.fingerprint,
        "checkpoint":checkpoint, "harness_version":harness.version, "partition":args.partition,
        "seeds":args.seeds, "selection":False})
    backend = CommandBackend(json.loads(args.backend.read_text()), args.output / "costs.jsonl")
    tasks = manifest.partitions[args.partition]  # Explicit final evaluator, never outer loop.
    rows = backend.evaluate(checkpoint, harness, tasks, tuple(args.seeds), args.output / "evaluation")
    if {r.key for r in rows} != {(task,seed) for task in tasks for seed in args.seeds}:
        raise ValueError("Final evaluator returned wrong tasks or seeds")
    result = {"benchmark":"appworld", "partition":args.partition, **aggregate_results(rows),
              "cost":asdict(sum((r.cost for r in rows), Cost()))}
    write_json(args.output / "aggregate.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
