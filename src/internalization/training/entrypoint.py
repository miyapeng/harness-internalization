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
    if default_stage is None: parser.add_argument("stage", choices=("propose", "target", "check_internalization", "evaluate", "train"))
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--device", help="Legacy requests only; must match resolved config if supplied")
    parser.add_argument("--benchmark", choices=("alfworld", "appworld", "terminalbench2", "swebench_pro", "hotpotqa", "webshop", "lawbench"), default="alfworld")
    parser.add_argument("--env-config", type=Path)
    parser.add_argument("--final-evaluation", action="store_true", help="Held-out evaluation only, never training or search")
    args = parser.parse_args()
    stage = default_stage or args.stage
    if args.final_evaluation and stage != "evaluate": raise ValueError("Final evaluation cannot train/propose")
    request = json.loads(args.request.read_text())
    if request["stage"] != stage: raise ValueError("Request stage mismatch")
    from ..core.execution_config import request_execution, resolve_execution, save_effective
    execution = request_execution(request)
    common = {"schema_version", "stage", "effective_config", "effective_config_hash"}
    fields = {
        "propose": {"checkpoint","harness","task_ids","cycle","candidate_count","trajectories","scores","history"},
        "target": {"checkpoint","harness","task_ids","cycle","candidate_count","trajectories","scores","history"},
        "evaluate": {"checkpoint","harness","task_ids","seeds"},
        "check_internalization": {"checkpoint","target","task_ids"},
        "train": {"teacher_checkpoint","student_checkpoint","full_harness","reduced_harness","target","task_ids","planned_update_batches","optimizer_steps","sampling_state"},
    }
    if set(request)-common-fields[stage]: raise ValueError("Unknown stage request fields")
    if execution is not None and args.device is not None and args.device != execution["device"]:
        raise ValueError("--device conflicts with effective configuration")
    if execution is not None and execution["mode"] == "evolution_only" and stage in ("target","check_internalization","train"):
        raise ValueError("evolution_only mode cannot enter internalization stages")
    out = args.response.resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    if args.response.exists(): raise FileExistsError(args.response)
    started = time.monotonic()
    if execution is not None:
        save_effective(out,execution)
        from ..core.sampling import seed_process
        seed_process(execution["seeds"]["model_sampling_seed"])
        write_json(out/"seed_config.json",execution["seeds"])
    if stage in ("propose","target"):
        proposer = APIProposer()
        proposal = ProposalRequest(request["checkpoint"], harness_from_dict(request["harness"]),
            tuple(request["task_ids"]), tuple(trajectory_from_dict(t) for t in request["trajectories"]),
            tuple(EpisodeResult(**{**r, "cost": Cost(**r["cost"])}) for r in request["scores"]),
            tuple(request["history"]), request["cycle"], request["candidate_count"], out)
        from ..harness.revision import HarnessRevision, RevisionStore
        if isinstance(proposal.harness,HarnessRevision):
            from ..evolution.code_proposer import CodeProposer
            proposer=CodeProposer(RevisionStore(Path(proposal.harness.path).parent,proposal.harness.policy),
                **(execution["proposer"] if execution else {}))
            if stage=="target":
                target=proposer.propose_target(proposal)
                result={"target":target.to_dict() if target else None,"cost":asdict(proposer.last_cost)}
            else:
                candidates=proposer.propose(proposal)
                result={"candidates":[c.to_dict() if hasattr(c,"to_dict") else c for c in candidates],"cost":asdict(proposer.last_cost)}
        else:
            if stage=="target": raise ValueError("Code target proposal requires a Harness revision")
            candidates = proposer.propose(proposal)
            result = {"candidate_sources": [c.source for c in candidates], "cost": asdict(proposer.last_cost)}
    else:
        model_options, policy_options, max_steps = {}, {}, 30
        if args.benchmark == "appworld":
            from ..benchmarks.appworld import AppWorldConfig, environment_factory
            config = AppWorldConfig.load(args.env_config or Path("configs/appworld.json"))
            if execution is not None:
                config = replace(config, max_steps=execution["max_steps"], **execution["model"])
            factory = environment_factory(config, out, training=stage == "train")
            model_options, max_steps = config.model_options, config.max_steps
            policy_options = {**model_options, "max_prompt_tokens":config.max_prompt_tokens}
            write_json(out / "benchmark_protocol.json", {"benchmark":"appworld", "config":asdict(config),
                "model_options":model_options, "policy_options":policy_options,
                "supervision":"same_behavior_policy_H_plus_minus_H_minus", "reward":"terminal_official_task_success"})
        elif args.benchmark in ("terminalbench2", "swebench_pro", "hotpotqa", "webshop", "lawbench"):
            from ..benchmarks.common import BenchmarkConfig, Catalog
            from ..benchmarks.isolated import environment_factory
            config = BenchmarkConfig.load(args.env_config or Path(f"configs/{args.benchmark}.json"))
            if config.benchmark != args.benchmark: raise ValueError("Benchmark configuration mismatch")
            catalog = Catalog(config.catalog, args.benchmark)
            catalog.check_selection(request["task_ids"], final=args.final_evaluation)
            config = replace(config, catalog=str(catalog.path), options={**config.options, "catalog_hash":catalog.fingerprint})
            if execution is not None:
                config = replace(config, max_steps=execution["max_steps"], **execution["model"])
            factory = environment_factory(config, out, training=stage == "train")
            model_options, max_steps = config.model_options, config.max_steps
            policy_options = {**model_options, "max_prompt_tokens":config.max_prompt_tokens}
            write_json(out/"benchmark_protocol.json", {"benchmark":args.benchmark, "config":asdict(config),
                "catalog_hash":catalog.fingerprint, "dataset_revision":catalog.revision,
                "final_evaluation":args.final_evaluation, "model_options":model_options,
                "supervision":"same_behavior_policy_H_plus_minus_H_minus"})
        else:
            factory = lambda: AlfworldEnvironment(args.env_config or Path("configs/alfworld.yaml"))
        if execution is None:
            # Historical direct requests get one explicit, recorded compatibility resolution.
            execution = resolve_execution({"device":args.device or "cuda:0"}, benchmark_limits={
                "max_steps":max_steps,"model":policy_options or model_options})
            save_effective(out,execution)
        model_options = execution["model"]
        policy_options = {**model_options, **execution["optimizer"]}
        runner = InteractionTaskRunner(factory, lambda p: FrozenHFBackend(p, device=execution["device"], **model_options),
                                       max_steps=execution["max_steps"], supervision=execution["supervision"],
                                       model_sampling_seed=execution["seeds"]["model_sampling_seed"],environment_seed=execution["seeds"]["environment_seed"])
        if stage == "check_internalization":
            from ..harness.revision import InternalizationTarget
            result=runner.check_internalization(request["checkpoint"],InternalizationTarget.from_dict(request["target"]),
                tuple(request["task_ids"]),output=out/"check")
        elif stage == "evaluate":
            trajectories = runner.rollout(request["checkpoint"], harness_from_dict(request["harness"]),
                tuple(request["task_ids"]), seeds=tuple(request["seeds"]), output=out)
            result = {"episodes": [asdict(r) for r in trajectories.evaluations],
                      "cost": asdict(sum((t.cost for t in trajectories.trajectories), Cost()))}
            if args.benchmark == "appworld":
                from ..benchmarks.appworld_manifest import aggregate_results
                result["benchmark_metrics"] = aggregate_results(trajectories.evaluations)
            elif args.benchmark in ("terminalbench2", "swebench_pro", "hotpotqa", "webshop", "lawbench"):
                from ..benchmarks.aggregate import aggregate
                result["benchmark_metrics"] = aggregate(config, out, trajectories.evaluations)
                result["cost"]["latency_s"] = time.monotonic()-started
        else:
            from .trainer import ModuleTrainer
            from .verl_backend import VerlPolicy
            from .module_advantage import AdvantageConfig
            trainer = ModuleTrainer(runner, lambda p: VerlPolicy(p, device=execution["device"], **policy_options),
                lambda p: FrozenHFBackend(p, device=execution["reference_device"], **model_options),
                config=AdvantageConfig(**execution["advantage"]), supervision=execution["supervision"],
                tasks_per_batch=execution["tasks_per_batch"], rollouts_per_task=execution["rollouts_per_task"],
                sampling_state=request.get("sampling_state",{}),
                sampling_seed=execution["seeds"]["run_seed"] if execution["schedule"]["profile"]=="budget_v1" else None,
                environment_seed=execution["seeds"]["environment_seed"])
            target=request["target"]
            if isinstance(target,dict):
                from ..harness.revision import InternalizationTarget
                target=InternalizationTarget.from_dict(target)
            from ..core.execution_config import NoActorUpdates
            budget=request.get("planned_update_batches", request.get("optimizer_steps"))
            if "planned_update_batches" in request and "optimizer_steps" in request:
                raise ValueError("Ambiguous batch budget; optimizer_steps is a legacy alias only")
            if type(budget) is not int or budget < 1: raise ValueError("Positive planned_update_batches required")
            if execution["mode"] != "internalization": raise ValueError("Training requires internalization mode")
            try:
                checkpoint = trainer.train(request["student_checkpoint"], request["teacher_checkpoint"],
                    harness_from_dict(request["full_harness"]), harness_from_dict(request["reduced_harness"]),
                    target=target, tasks=tuple(request["task_ids"]), budget=budget, output=out)
            except NoActorUpdates:
                checkpoint=request["student_checkpoint"]
            cost = asdict(trainer.last_cost)
            cost["latency_s"] = time.monotonic()-started
            result = {**trainer.last_summary, "checkpoint": checkpoint, "cost": cost}

    write_json(args.response, {"schema_version": 1, **result})


if __name__ == "__main__": main()
