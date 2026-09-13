#!/usr/bin/env python3
"""Explicit final evaluation, outside candidate selection and model training."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from internalization.benchmarks.common import BENCHMARKS
from internalization.command_backend import CommandBackend
from internalization.core.manifest import TaskManifest
from internalization.core.serialization import harness_from_dict
from internalization.core.types import Cost, write_json
from internalization.harness.module import Harness


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state", type=Path, help="Accepted cycle state with residual Harness")
    parser.add_argument("--partition", default="test")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0,1,2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = TaskManifest.load(args.manifest)
    if manifest.benchmark not in BENCHMARKS: raise ValueError("Use the existing evaluator for this benchmark")
    if len(set(args.seeds)) != len(args.seeds): raise ValueError("Duplicate seeds")
    config = json.loads(args.backend.read_text())
    if config.get("benchmark") != manifest.benchmark: raise ValueError("Backend/manifest benchmark mismatch")
    config["evaluate"] = config["evaluate"] + ["--final-evaluation"]
    checkpoint, harness = str(args.checkpoint.resolve(strict=True)), Harness()
    if args.state:
        state = json.loads(args.state.read_text())
        if str(Path(state["checkpoint"]).resolve(strict=True)) != checkpoint:
            raise ValueError("Accepted state checkpoint mismatch")
        harness = harness_from_dict({"version":state["harness_version"], "modules":state["active_modules"]})
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output/"protocol.json", {"manifest_hash":manifest.fingerprint, "checkpoint":checkpoint,
        "harness_version":harness.version, "partition":args.partition, "seeds":args.seeds, "selection":False})
    backend = CommandBackend(config, args.output/"costs.jsonl")
    tasks = manifest.partitions[args.partition]
    rows = backend.evaluate(checkpoint, harness, tasks, tuple(args.seeds), args.output/"evaluation")
    if len(rows) != len(tasks)*len(args.seeds) or {r.key for r in rows} != {(t,s) for t in tasks for s in args.seeds}:
        raise ValueError("Wrong final evaluation task/seed coverage")
    response = json.loads((args.output/"evaluation"/"response.json").read_text())
    result = {"benchmark":manifest.benchmark, "partition":args.partition,
        "benchmark_metrics":response["benchmark_metrics"], "cost":response["cost"]}
    write_json(args.output/"aggregate.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
