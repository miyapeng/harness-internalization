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
    run.add_argument("--experiment-config", type=Path, help="Strict execution settings; replace backend execution settings")
    run.add_argument("--checkpoint", help="Required for a fresh run; optional check when resuming --state")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--planned-update-batches", "--train-steps", dest="train_steps", type=int,
                     help="Total planned update batches (not optimizer steps); default 300, or resumed protocol")
    run.add_argument("--cycles", type=int, help="Total planned cycles; default 3, or preserved from resumed protocol")
    code_source=run.add_mutually_exclusive_group()
    code_source.add_argument("--harness-workspace",type=Path,help="Import a configured executable Harness source tree")
    code_source.add_argument("--harness-revision",type=Path,help="JSON HarnessRevision or accepted state/deployment")
    code_source.add_argument("--state",type=Path,help="Continue from an accepted checkpoint/Harness/protocol pair")
    run.add_argument("--protocol",type=Path,help="Protocol sidecar for a relocated historical state")
    run.add_argument("--revision-store",type=Path,default=Path("runs/harness-revisions"))
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
        from .evaluation.retirement import RetirementPolicy
        from .core.accepted_state import load_accepted_state
        config = json.loads(args.backend.read_text())
        if args.experiment_config:
            config["execution"] = json.loads(args.experiment_config.read_text())
        backend = CommandBackend(config, args.output.parent / f"{args.output.name}.costs.jsonl")
        if args.output.exists(): raise FileExistsError(args.output)
        manifest=TaskManifest.load(args.manifest)
        accepted=None
        raw_revision=None
        state_path=args.state
        if args.harness_revision:
            raw_revision=json.loads(args.harness_revision.read_text())
            if raw_revision.get("format")!="code_revision_v1": state_path=args.harness_revision
        if state_path:
            accepted=load_accepted_state(state_path,manifest,checkpoint=args.checkpoint,protocol_path=args.protocol)
            if "loop" not in accepted.protocol: raise ValueError("Accepted state has no resumable loop protocol")
        elif args.protocol: raise ValueError("--protocol requires an accepted --state")
        options=dict(accepted.protocol["loop"]) if accepted else {}
        if args.cycles is not None: options["cycles"]=args.cycles
        if args.train_steps is not None: options["total_train_steps"]=args.train_steps
        if "seeds" in options: options["seeds"]=tuple(options["seeds"])
        loop=LoopConfig(**options)
        policy=RetirementPolicy(**accepted.protocol["retirement"]) if accepted else RetirementPolicy()
        attribution_policy=(AttributionPolicy.load(args.attribution_policy) if args.attribution_policy else
            AttributionPolicy(**accepted.protocol.get("attribution",{})) if accepted else AttributionPolicy())
        initial_harness=None
        from .harness.revision import HarnessRevision
        versioned=bool(args.harness_workspace or (raw_revision and not state_path) or
            (accepted and isinstance(accepted.harness,HarnessRevision)))
        manifest.validate_loop(loop.cycles,versioned=versioned,
            cohort_minimum=max(policy.min_tasks,attribution_policy.min_tasks) if versioned else None)
        if accepted:
            accepted.check_resume(manifest,loop,policy,attribution_policy)
            checkpoint=accepted.checkpoint
        else:
            if not args.checkpoint: raise ValueError("A fresh run requires --checkpoint")
            checkpoint=str(Path(args.checkpoint).resolve(strict=True))
        if args.harness_workspace:
            from .harness.revision import RevisionStore
            initial_harness=RevisionStore(args.revision_store).import_directory(args.harness_workspace)
        elif raw_revision and not state_path: initial_harness=HarnessRevision.from_dict(raw_revision)
        result = run_outer_loop(backend,manifest,checkpoint,args.output,loop,policy,
                  attribution_policy=attribution_policy,initial_harness=initial_harness,accepted_state=accepted)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
