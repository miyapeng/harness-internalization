"""Standalone process entrypoints for project components."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

from ..core.serialization import harness_from_dict, trajectory_from_dict
from ..core.types import Cost, EpisodeResult, write_json
from ..core.interfaces import ProposalRequest
from ..benchmarks.alfworld import AlfworldEnvironment
from ..evolution.proposer import APIProposer
from .rollout import InteractionTaskRunner
from .teacher_backend import FrozenHFBackend


def main(default_stage=None):
    parser = argparse.ArgumentParser()
    if default_stage is None: parser.add_argument("stage", choices=("propose", "evaluate", "train"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--device", default=os.environ.get("HI_DEVICE", "cuda:0"))
    parser.add_argument("--benchmark", choices=("alfworld", "appworld", "terminalbench2", "swebench_pro", "hotpotqa", "lawbench"), default="alfworld")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--final-evaluation", action="store_true", help="Held-out evaluation only, never training or search")
    args = parser.parse_args()
    stage = default_stage or args.stage
    if args.final_evaluation and stage != "evaluate": raise ValueError("Final evaluation cannot train/propose")
    request = json.loads(args.request.read_text())
    if request["stage"] != stage: raise ValueError("Request stage mismatch")
    out = args.response.resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    if args.response.exists(): raise FileExistsError(args.response)
    started = time.monotonic()
    if stage == "propose":
        proposer = APIProposer()
        proposal = ProposalRequest(request["checkpoint"], harness_from_dict(request["harness"]),
            tuple(request["task_ids"]), tuple(trajectory_from_dict(t) for t in request["trajectories"]),
            tuple(EpisodeResult(**{**r, "cost": Cost(**r["cost"])}) for r in request["scores"]),
            tuple(request["history"]), request["cycle"], request["candidate_count"], out)
        candidates = proposer.propose(proposal)
        result = {"candidate_sources": [c.source for c in candidates], "cost": asdict(proposer.last_cost)}
    else:
        model_options, policy_options, max_steps = {}, {}, 30
        if args.benchmark == "appworld":
            from ..benchmarks.appworld import AppWorldConfig, environment_factory
            config = AppWorldConfig.load(args.env_config or Path("configs/appworld.json"))
            factory = environment_factory(config, out, training=stage == "train")
            model_options, max_steps = config.model_options, config.max_steps
            policy_options = {**model_options, "max_prompt_tokens":config.max_prompt_tokens}
            write_json(out / "benchmark_protocol.json", {"benchmark":"appworld", "config":asdict(config),
                "model_options":model_options, "policy_options":policy_options,
                "supervision":"same_behavior_policy_H_plus_minus_H_minus", "reward":"terminal_official_task_success"})
        elif args.benchmark in ("terminalbench2", "swebench_pro", "hotpotqa", "lawbench"):
            from ..benchmarks.common import BenchmarkConfig, Catalog
            from ..benchmarks.isolated import environment_factory
            config = BenchmarkConfig.load(args.env_config or Path(f"configs/{args.benchmark}.json"))
            if config.benchmark != args.benchmark: raise ValueError("Benchmark configuration mismatch")
            catalog = Catalog(config.catalog, args.benchmark)
            catalog.check_selection(request["task_ids"], final=args.final_evaluation)
            config = replace(config, catalog=str(catalog.path), options={**config.options, "catalog_hash":catalog.fingerprint})
            factory = environment_factory(config, out, training=stage == "train")
            model_options, max_steps = config.model_options, config.max_steps
            policy_options = {**model_options, "max_prompt_tokens":config.max_prompt_tokens}
            write_json(out/"benchmark_protocol.json", {"benchmark":args.benchmark, "config":asdict(config),
                "catalog_hash":catalog.fingerprint, "dataset_revision":catalog.revision,
                "final_evaluation":args.final_evaluation, "model_options":model_options,
                "supervision":"same_behavior_policy_H_plus_minus_H_minus"})
        else:
            factory = lambda: AlfworldEnvironment(args.env_config or Path("configs/alfworld.yaml"))
        runner = InteractionTaskRunner(factory, lambda p: FrozenHFBackend(p, device=args.device, **model_options),
                                       max_steps=max_steps)
        if stage == "evaluate":
            trajectories = runner.rollout(request["checkpoint"], harness_from_dict(request["harness"]),
                tuple(request["task_ids"]), seeds=tuple(request["seeds"]), output=out)
            result = {"episodes": [asdict(r) for r in trajectories.evaluations],
                      "cost": asdict(sum((t.cost for t in trajectories.trajectories), Cost()))}
            if args.benchmark == "appworld":
                from ..benchmarks.appworld_manifest import aggregate_results
                result["benchmark_metrics"] = aggregate_results(trajectories.evaluations)
            elif args.benchmark in ("terminalbench2", "swebench_pro", "hotpotqa", "lawbench"):
                from ..benchmarks.aggregate import aggregate
                result["benchmark_metrics"] = aggregate(config, out, trajectories.evaluations)
                result["cost"]["latency_s"] = time.monotonic()-started
        else:
            from .trainer import ModuleTrainer
            from .verl_backend import VerlPolicy
            trainer = ModuleTrainer(runner, lambda p: VerlPolicy(p, device=args.device, **policy_options),
                lambda p: FrozenHFBackend(p, device=os.environ.get("HI_TEACHER_DEVICE", "cpu"), **model_options))
            checkpoint = trainer.train(request["student_checkpoint"], request["teacher_checkpoint"],
                harness_from_dict(request["full_harness"]), harness_from_dict(request["reduced_harness"]),
                target=request["target"], tasks=tuple(request["task_ids"]),
                budget=request["optimizer_steps"], output=out)
            cost = asdict(trainer.last_cost)
            cost["latency_s"] = time.monotonic()-started
            result = {"checkpoint": checkpoint, "optimizer_steps_completed": request["optimizer_steps"],
                      "training_batches_completed": request["optimizer_steps"], "cost": cost}
    write_json(args.response, {"schema_version": 1, **result})


if __name__ == "__main__": main()
