"""Accept a runnable improvement first; internalization of a selected difference is optional."""
from dataclasses import asdict, replace

from .core.interfaces import ProposalRequest
from .core.types import Journal, write_json
from .evolution.search import evaluate_tasks
from .evolution.revision_search import search_revisions, public_history
from .evaluation.attribution import evaluate_attribution
from .evaluation.retirement import PairedRetirementEvaluator
from .harness.revision import InternalizationTarget
from .core.accepted_state import AcceptedAgentState


def run_revision_loop(components,manifest,checkpoint,initial_harness,output,config,policy,attribution_policy,*,accepted_state=None):
    manifest.validate_loop(config.cycles, versioned=True,
        cohort_minimum=max(policy.min_tasks,attribution_policy.min_tasks))
    if accepted_state is not None: accepted_state.check_resume(manifest,config,policy,attribution_policy)
    start_cycle=accepted_state.next_cycle if accepted_state is not None else 0
    train,search,dev=(manifest.partition(p) for p in ("train","search","dev"))
    cohorts=[manifest.partition(f"retirement_{i}") for i in range(config.cycles)]
    acceptance=[manifest.partition(f"acceptance_{i}") for i in range(config.cycles)]
    output.mkdir(parents=True,exist_ok=False)
    journal=Journal(output/"events.jsonl")
    harness=initial_harness
    evaluator=components.retirement or PairedRetirementEvaluator(components.runner,policy)
    protocol=accepted_state.protocol if accepted_state is not None else {"mode":"versioned_code_selective_internalization","loop":asdict(config),
        "retirement":asdict(policy),"attribution":asdict(attribution_policy),"harness_acceptance":asdict(attribution_policy),
        "manifest_hash":manifest.fingerprint,"initial_checkpoint":checkpoint,"initial_harness":harness.to_dict()}
    write_json(output/"protocol.json",protocol)
    initial=accepted_state or AcceptedAgentState(checkpoint,harness,manifest.fingerprint,protocol,0)
    write_json(output/"initial_agent.json",initial.to_dict())
    archive=[]

    def save(folder,reason,*,candidate=None,target=None,model_decision="unchanged",module_decision="retain",**extra):
        entry={**AcceptedAgentState(checkpoint,harness,manifest.fingerprint,protocol,cycle+1).to_dict(),
            "cycle":cycle,"reason":reason,
            "model_decision":model_decision,"module_decision":module_decision,
            "candidate_id":candidate.candidate_id if candidate else None,
            "target":target.to_dict() if target else None,**extra}
        archive.append(entry)
        write_json(folder/"state.json",entry)
        journal.append("cycle_complete",**entry)

    for cycle in range(start_cycle,config.cycles):
        folder=output/f"cycle_{cycle:02d}"
        folder.mkdir()
        parent=harness
        base_search=evaluate_tasks(components.runner,checkpoint,parent,search,config.seeds,folder/"baseline_search")
        base_dev=evaluate_tasks(components.runner,checkpoint,parent,dev,config.seeds,folder/"baseline_dev")
        request=ProposalRequest(checkpoint,parent,search,base_search.trajectories,base_search.evaluations,
            public_history(journal),cycle,config.candidates_per_cycle,folder/"proposals")
        candidate=search_revisions(components,request,baseline_search=base_search,baseline_dev=base_dev,
            dev_tasks=dev,seeds=config.seeds,policy=policy,journal=journal)
        if candidate is None:
            save(folder,"no_useful_candidate",module_decision="unchanged")
            continue
        full=candidate.full_revision
        try:
            plus=evaluate_tasks(components.runner,checkpoint,full,acceptance[cycle],config.seeds,folder/"acceptance_plus")
            before=evaluate_tasks(components.runner,checkpoint,parent,acceptance[cycle],config.seeds,folder/"acceptance_parent")
            gate=evaluate_attribution(plus.evaluations,before.evaluations,attribution_policy)
            write_json(folder/"harness_acceptance.json",{**gate,"parent_revision":parent.version,"full_revision":full.version})
        except Exception as exc:
            save(folder,"harness_acceptance_failed",candidate=candidate,error=str(exc));continue
        if not gate["passed"]:
            save(folder,"harness_acceptance_failed",candidate=candidate);continue
        harness=full  # Durable acceptance precedes any attempt to construct H-minus.
        write_json(folder/"accepted_harness.json",AcceptedAgentState(checkpoint,harness,manifest.fingerprint,protocol).to_dict())
        target=None
        try:
            provider=components.targets or (components.proposer if hasattr(components.proposer,"propose_target") else None)
            if provider is not None:
                target=provider.propose_target(replace(request,harness=full,output=folder/"target_proposal",count=1))
            if target is None:
                save(folder,"accepted_without_internalization",candidate=candidate,detail="no_reduction_proposed");continue
            if not isinstance(target,InternalizationTarget):
                target=None
                raise ValueError("unsupported: expected at most one executable InternalizationTarget")
            if target.full_revision.version!=full.version: raise ValueError("Target points to a different accepted revision")
            write_json(folder/"internalization_target.json",target.to_dict())
            target.validate_structure()
            check=getattr(components.runner,"check_internalization",None)
            if check is None: raise ValueError("unsupported: runner has no executable supervision checker")
            compatibility=check(checkpoint,target,search,output=folder/"compatibility")
            if not compatibility["supported"]: raise ValueError(compatibility["reason"])
        except Exception as exc:
            save(folder,"accepted_without_internalization",candidate=candidate,target=target,detail=str(exc));continue
        reduced=target.reduced_revision
        try:
            a=evaluate_tasks(components.runner,checkpoint,full,cohorts[cycle],config.seeds,folder/"A")
            b=evaluate_tasks(components.runner,checkpoint,reduced,cohorts[cycle],config.seeds,folder/"B")
            gate=evaluate_attribution(a.evaluations,b.evaluations,attribution_policy)
            write_json(folder/"attribution.json",{**gate,"full_revision":full.version,"reduced_revision":reduced.version})
        except Exception as exc:
            save(folder,"attribution_failed",candidate=candidate,target=target,detail=str(exc));continue
        if not gate["passed"]:
            save(folder,"attribution_failed",candidate=candidate,target=target);continue
        before_checkpoint=checkpoint
        try:
            proposed=components.trainer.train(checkpoint,checkpoint,full,reduced,None,target=target,tasks=train,
                budget=config.total_train_steps//config.cycles,output=folder/"training")
            if proposed==checkpoint: raise ValueError("Training must create a new checkpoint")
            verdict=evaluator.evaluate(checkpoint,proposed,full,reduced,tasks=cohorts[cycle],seeds=config.seeds,
                output=folder,before_cells={"A":list(a.evaluations),"B":list(b.evaluations)})
            decisions=(verdict.get("model_decision"),verdict.get("module_decision"))
            if decisions not in (("accept","retire"),("accept","retain"),("rollback","retain")):
                raise ValueError("Invalid independent model/module decisions")
        except Exception as exc:
            save(folder,"training_or_audit_failed",candidate=candidate,target=target,
                model_decision="rollback",detail=str(exc),before_checkpoint=before_checkpoint);continue
        checkpoint=proposed if verdict["model_decision"]=="accept" else before_checkpoint
        harness=reduced if verdict["module_decision"]=="retire" else full
        write_json(folder/"retirement.json",verdict)
        save(folder,"internalization_audited",candidate=candidate,target=target,
            model_decision=verdict["model_decision"],module_decision=verdict["module_decision"],
            before_checkpoint=before_checkpoint,proposed_checkpoint=proposed)
    result={**AcceptedAgentState(checkpoint,harness,manifest.fingerprint,protocol,config.cycles).to_dict(),"archive":archive}
    write_json(output/"deployment.json",result)
    return result
