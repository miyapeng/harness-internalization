"""Experiment orchestration depending only on project interfaces and types."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from .core.interfaces import Components, ProposalRequest
from .core.adapters import components_for
from .core.manifest import TaskManifest
from .core.types import Journal, write_json
from .harness.module import Harness
from .evolution.archive import CandidateArchive
from .evolution.search import evaluate_tasks, search_candidates
from .evaluation.retirement import RetirementPolicy, PairedRetirementEvaluator
from .evaluation.attribution import AttributionPolicy, evaluate_attribution

@dataclass(frozen=True)
class LoopConfig:
    cycles: int = 3
    candidates_per_cycle: int = 2
    total_train_steps: int = 300
    seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self):
        if self.cycles < 1 or self.candidates_per_cycle < 1 or self.total_train_steps < self.cycles:
            raise ValueError("Invalid loop budget")
        if self.total_train_steps % self.cycles:
            raise ValueError("Training budget must divide equally across cycles")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Unique evaluation seeds required")


def run_outer_loop(backend, manifest: TaskManifest, checkpoint: str,
                   output: Path, config=LoopConfig(), policy=RetirementPolicy(), *,
                   attribution_policy=AttributionPolicy(), initial_harness=None):
    components = components_for(backend)
    if initial_harness is not None:
        from .harness.revision import HarnessRevision
        from .revision_loop import run_revision_loop
        if not isinstance(initial_harness,HarnessRevision): raise TypeError("Expected an explicit runnable HarnessRevision")
        return run_revision_loop(components,manifest,checkpoint,initial_harness,output,config,policy,attribution_policy)
    manifest.validate()
    train, search, dev = (manifest.partition(p) for p in ("train", "search", "dev"))
    cohorts = [manifest.partition(f"retirement_{k}") for k in range(config.cycles)]
    output.mkdir(parents=True, exist_ok=False)
    journal = Journal(output / "events.jsonl")
    candidate_archive = CandidateArchive(output / "candidate_archive.jsonl")
    evaluator = components.retirement or PairedRetirementEvaluator(components.runner, policy)
    write_json(output / "protocol.json", {"loop": asdict(config), "retirement": asdict(policy),
               "manifest_hash": manifest.fingerprint, "initial_checkpoint": checkpoint})
    # Immutable artifact written before any rollout/candidate evaluation.
    write_json(output / "attribution_policy.json", asdict(attribution_policy))
    harness, archive = Harness(), []
    for cycle in range(config.cycles):
        folder = output / f"cycle_{cycle:02d}"
        folder.mkdir()
        base_search = evaluate_tasks(components.runner, checkpoint, harness, search, config.seeds, folder / "baseline_search")
        request = ProposalRequest(checkpoint, harness, search, base_search.trajectories,
            base_search.evaluations, candidate_archive.proposer_history(), cycle, config.candidates_per_cycle,
            folder / "proposals")
        base_dev = evaluate_tasks(components.runner, checkpoint, harness, dev, config.seeds, folder / "baseline_dev")
        candidate = search_candidates(components.proposer, components.runner, request, dev_tasks=dev,
            baseline_search=base_search, baseline_dev=base_dev, seeds=config.seeds, policy=policy,
            archive=candidate_archive, journal=journal)
        if candidate is None:
            journal.append("cycle_skipped", cycle=cycle, reason="No useful candidate; budget unspent")
            continue
        module = candidate.module()
        full = Harness(harness.modules + (module,))
        reduced = full.without(module.name)
        write_json(folder / "phase.json", {"teacher_checkpoint": checkpoint, "full": full.version,
                   "reduced": reduced.version, "target": module.name, "module_source": module.source})
        tasks = cohorts[cycle]
        a = evaluate_tasks(components.runner, checkpoint, full, tasks, config.seeds, folder / "A")
        b = evaluate_tasks(components.runner, checkpoint, reduced, tasks, config.seeds, folder / "B")
        attribution = evaluate_attribution(a.evaluations, b.evaluations, attribution_policy)
        evidence = str((folder / "attribution.json").relative_to(output))
        write_json(folder / "attribution.json", {**attribution, "checkpoint": checkpoint,
            "candidate_id": candidate.candidate_id, "parent": candidate.parent,
            "module": module.name, "module_version": module.version,
            "h_plus": full.version, "h_minus": reduced.version,
            "manifest_hash": manifest.fingerprint, "partition": f"retirement_{cycle}",
            "seeds": config.seeds, "evaluations": {"A": "A/episodes.json", "B": "B/episodes.json"}})
        candidate_archive.record_attribution(candidate, result=attribution, checkpoint=checkpoint,
            h_plus=full, h_minus=reduced, evidence=evidence)
        if not attribution["passed"]:
            archive.append({"cycle": cycle, "module": module.name, "version": module.version,
                "source": module.source, "candidate_id": candidate.candidate_id, "parent": candidate.parent,
                "decision": "discard", "reason": "no_external_contribution", "attribution": evidence})
            write_json(folder / "state.json", {"checkpoint": checkpoint, "harness_version": harness.version,
                "active_modules": [{"name": m.name, "version": m.version, "source": m.source}
                                   for m in harness.modules], "archive": archive})
            journal.append("cycle_skipped", cycle=cycle, reason="no_external_contribution",
                candidate_id=candidate.candidate_id, checkpoint=checkpoint,
                attribution=evidence, training_batches_spent=0)
            continue
        new_checkpoint = components.trainer.train(checkpoint, checkpoint, full, reduced, None,
            target=module.name, tasks=train, budget=config.total_train_steps // config.cycles, output=folder / "training")
        if new_checkpoint == checkpoint:
            raise ValueError("Training must produce a new checkpoint, preserving the teacher snapshot")
        verdict = evaluator.evaluate(checkpoint, new_checkpoint, full, reduced, tasks=tasks,
            seeds=config.seeds, output=folder, before_cells={"A": list(a.evaluations), "B": list(b.evaluations)})
        # Custom evaluators must explicitly decide both; never silently accept
        # a checkpoint on the strength of a legacy retire/retain verdict.
        model_decision = verdict.get("model_decision")
        module_decision = verdict.get("module_decision")
        if (model_decision, module_decision) not in (
                ("accept", "retire"), ("accept", "retain"), ("rollback", "retain")):
            raise ValueError("Evaluator must return valid independent model/module decisions")
        before_checkpoint = checkpoint
        checkpoint = new_checkpoint if model_decision == "accept" else before_checkpoint
        harness = reduced if module_decision == "retire" else full
        outcome = {"model_decision": model_decision, "module_decision": module_decision,
                   "before_checkpoint": before_checkpoint, "proposed_checkpoint": new_checkpoint,
                   "accepted_checkpoint": checkpoint}
        verdict = {**verdict, **outcome}
        write_json(folder / "retirement.json", verdict)
        candidate_archive.record_outcome(candidate, result=verdict,
            evidence=str((folder / "retirement.json").relative_to(output)))
        archive.append({"cycle": cycle, "module": module.name, "version": module.version,
                        "source": module.source, "decision": module_decision, **outcome})
        write_json(folder / "state.json", {"checkpoint": checkpoint, **outcome,
            "harness_version": harness.version,
            "active_modules": [{"name": m.name, "version": m.version, "source": m.source}
                               for m in harness.modules], "archive": archive})
        journal.append("cycle_complete", cycle=cycle, checkpoint=checkpoint,
                       active_modules=[m.name for m in harness.modules], decision=module_decision, **outcome)
    result = {"checkpoint": checkpoint, "active_modules": [m.name for m in harness.modules],
              "harness_version": harness.version, "archive": archive}
    write_json(output / "deployment.json", result)
    return result
