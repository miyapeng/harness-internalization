from __future__ import annotations

import argparse
import json
from pathlib import Path

from .records import read_results, write_json


def main():
    parser = argparse.ArgumentParser(prog="hi")
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Synthetic CPU wiring test; not benchmark evidence")
    demo.add_argument("--output", type=Path, required=True)
    retirement = commands.add_parser("retirement")
    for cell in "ABCD": retirement.add_argument(f"--{cell}", type=Path, required=True)
    retirement.add_argument("--output", type=Path, required=True)
    retirement.add_argument("--margin", type=float, default=0.02)
    validate = commands.add_parser("validate-module")
    validate.add_argument("path", type=Path)
    run = commands.add_parser("run")
    run.add_argument("--manifest", type=Path, required=True)
    run.add_argument("--backend", type=Path, required=True)
    run.add_argument("--checkpoint", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--train-steps", type=int, default=300)
    run.add_argument("--attribution-policy", type=Path,
                     help="Fixed pre-training gate JSON; defaults to 95%% task bootstrap, lower gain > 0")
    phase = commands.add_parser("prepare-phase")
    phase.add_argument("--checkpoint", type=Path, required=True)
    phase.add_argument("--module", type=Path, required=True)
    phase.add_argument("--manifest", type=Path, required=True)
    phase.add_argument("--output", type=Path, required=True)
    phase.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "demo":
        from .demo import run_demo
        result = run_demo(args.output)
    elif args.command == "retirement":
        from .retirement_eval import RetirementPolicy, evaluate_retirement
        result = evaluate_retirement({c: read_results(getattr(args, c)) for c in "ABCD"},
                                     RetirementPolicy(performance_margin=args.margin))
        write_json(args.output, result)
    elif args.command == "validate-module":
        from .harness_modules import ControlModule
        module = ControlModule.load(args.path)
        result = {"name": module.name, "kind": module.kind, "version": module.version}
    elif args.command == "prepare-phase":
        from .backends import checkpoint_fingerprint
        from .harness_modules import ControlModule, Harness
        from .manifest import TaskManifest
        module = ControlModule.load(args.module)
        manifest = TaskManifest.load(args.manifest)
        result = {"teacher_checkpoint": str(args.checkpoint.resolve()),
                  "teacher_fingerprint": checkpoint_fingerprint(args.checkpoint),
                  "teacher_device": args.device, "full_modules": [str(args.module.resolve())],
                  "full_harness_hash": Harness((module,)).version, "target": module.name,
                  "train_task_ids": manifest.partition("train"), "manifest_hash": manifest.fingerprint,
                  "supervision": "targeted", "max_context": 8192, "max_new_tokens": 192,
                  "rollout_trace_path": str(args.output.resolve().parent / "rollout_cost.jsonl"),
                  "trace_path": str(args.output.resolve().parent / "teacher_trace.jsonl")}
        write_json(args.output, result)
    else:
        from .command_backend import CommandBackend
        from .manifest import TaskManifest
        from .outer_loop import LoopConfig, run_outer_loop
        from .evaluation.attribution import AttributionPolicy
        if args.output.exists(): raise FileExistsError(args.output)
        config = json.loads(args.backend.read_text())
        backend = CommandBackend(config, args.output.parent / f"{args.output.name}.costs.jsonl")
        attribution_policy = AttributionPolicy.load(args.attribution_policy) if args.attribution_policy else AttributionPolicy()
        result = run_outer_loop(backend, TaskManifest.load(args.manifest), str(Path(args.checkpoint).resolve(strict=True)),
                  args.output, LoopConfig(total_train_steps=args.train_steps), attribution_policy=attribution_policy)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
