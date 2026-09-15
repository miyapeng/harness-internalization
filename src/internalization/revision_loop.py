"""Accept a runnable improvement first; internalization of a selected difference is optional."""
from dataclasses import asdict, replace

from .core.interfaces import ProposalRequest
from .core.types import Journal, write_json, digest
from .evolution.search import evaluate_tasks
from .evolution.revision_search import search_revisions, public_history, dev_acceptance_policy
from .evaluation.attribution import evaluate_attribution
from .evaluation.retirement import PairedRetirementEvaluator
from .harness.revision import InternalizationTarget
from .core.accepted_state import AcceptedAgentState
from .core.execution_config import NoActorUpdates


def run_revision_loop(components,manifest,checkpoint,initial_harness,output,config,policy,attribution_policy,*,accepted_state=None):
    execution = components.execution_config
    budget_mode = bool(execution and execution["schedule"]["profile"] == "budget_v1")
    if budget_mode:
        from .core.sampling import validate_budget_manifest, search_schedule
        validate_budget_manifest(manifest)
        if (config.cycles,config.candidates_per_cycle,config.total_train_steps,config.seeds) != (
                3,2,300,(execution["seeds"]["environment_seed"],)):
            raise ValueError("Loop configuration does not match budget_v1")
        if components.sampling_state is None: components.sampling_state = {}
        if accepted_state is not None:
            if accepted_state.sampling_state is None: raise ValueError("Resume requires persisted training sampler")
            components.sampling_state.update(accepted_state.sampling_state)
    acceptance_policy = dev_acceptance_policy(policy,budget=budget_mode)
    manifest.validate_loop(config.cycles, versioned=True,
        cohort_minimum=max(policy.min_tasks,attribution_policy.min_tasks))
    if accepted_state is not None: accepted_state.check_resume(manifest,config,policy,attribution_policy)
    from .core.seed_harnesses import bind_initial_seed
    initial_seed = bind_initial_seed(manifest.benchmark,initial_harness) if accepted_state is None else None
    start_cycle=accepted_state.next_cycle if accepted_state is not None else 0
    train,search,dev=(manifest.partition(p) for p in ("train","search","dev"))
    cohorts=[manifest.partition(f"retirement_{i}") for i in range(config.cycles)]
    output.mkdir(parents=True,exist_ok=False)
    journal=Journal(output/"events.jsonl")
    harness=initial_harness
    evaluator=components.retirement or PairedRetirementEvaluator(components.runner,policy)
    protocol=accepted_state.protocol if accepted_state is not None else {"mode":"versioned_code_selective_internalization","loop":asdict(config),
        "retirement":asdict(policy),"attribution":asdict(attribution_policy),"harness_acceptance":acceptance_policy,
        "manifest_hash":manifest.fingerprint,"initial_checkpoint":checkpoint,"initial_harness":harness.to_dict(),
        **({"initial_seed":initial_seed} if initial_seed is not None else {})}
    if protocol["harness_acceptance"] != acceptance_policy:
        # The user-authorized acceptance change applies only to future cycles;
        # retain the original snapshot and never rewrite any historical artifact.
        protocol={**protocol,"harness_acceptance":acceptance_policy,
            "acceptance_transition":{"from_protocol_hash":digest(protocol),
                "decision_source":"dev","effective_from_cycle":start_cycle}}
    execution = components.execution_config
    if execution is not None:
        from .core.execution_config import config_hash, save_effective
        identity = config_hash(execution)
        if accepted_state is not None and protocol.get("effective_config_hash") != identity:
            raise ValueError("Resumed execution configuration differs or historical configuration is unverified")
        protocol = {**protocol, "effective_config":execution, "effective_config_hash":identity}
        save_effective(output, execution)
    write_json(output/"protocol.json",protocol)
    initial=accepted_state or AcceptedAgentState(checkpoint,harness,manifest.fingerprint,protocol,0)
    write_json(output/"initial_agent.json",initial.to_dict())
    archive=[]

    def save(folder,reason,*,candidate=None,target=None,model_decision="unchanged",module_decision="retain",**extra):
        entry={**AcceptedAgentState(checkpoint,harness,manifest.fingerprint,protocol,cycle+1,components.sampling_state).to_dict(),
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
        if budget_mode:
            search = search_schedule(manifest.partition("search"),execution["seeds"]["run_seed"])[cycle]
            write_json(folder/"search_tasks.json",{"task_ids":list(search),"pool_size":96,
                "run_seed":execution["seeds"]["run_seed"],"cycle":cycle,"rule":"disjoint_seeded_queue"})
        base_search=evaluate_tasks(components.runner,checkpoint,parent,search,config.seeds,folder/"baseline_search")
        base_dev=None if budget_mode else evaluate_tasks(components.runner,checkpoint,parent,dev,config.seeds,folder/"baseline_dev")
        traces = base_search.trajectories
        if budget_mode:
            # Alternate poorest/best scores; all eight scores and every raw trace remain archived.
            ordered=sorted(traces,key=lambda t:(t.success,t.task_id,t.episode_id))
            representative=[]
            while ordered and len(representative)<4:
                representative.append(ordered.pop(0 if len(representative)%2==0 else -1))
            traces=tuple(representative)
        request=ProposalRequest(checkpoint,parent,search,traces,base_search.evaluations,
            public_history(journal),cycle,config.candidates_per_cycle,folder/"proposals")
        selection=search_revisions(components,request,baseline_search=base_search,baseline_dev=base_dev,
            dev_tasks=dev,seeds=config.seeds,policy=policy,journal=journal)
        if selection is None:
            write_json(folder/"harness_acceptance.json",{"decision_source":"dev","passed":False,
                "reason":"no_evaluable_candidate","checkpoint":checkpoint,"parent_revision":parent.version,
                "additional_evaluation_calls":0})
            save(folder,"no_useful_candidate",module_decision="unchanged")
            continue
        candidate=selection.candidate
        if selection.checkpoint != checkpoint or selection.parent.version != parent.version:
            raise ValueError("Dev selection does not belong to the current model/parent pair")
        write_json(folder/"harness_acceptance.json",selection.acceptance_record())
        if not selection.accepted:
            save(folder,"no_useful_candidate",candidate=candidate,module_decision="unchanged")
            continue
        full=candidate.full_revision
        harness=full  # Durable acceptance precedes any attempt to construct H-minus.
        write_json(folder/"accepted_harness.json",AcceptedAgentState(checkpoint,harness,manifest.fingerprint,protocol).to_dict())
        if execution is not None and execution["mode"] == "evolution_only":
            save(folder,"accepted_without_internalization",candidate=candidate,detail="evolution_only");continue
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
        except NoActorUpdates as exc:
            save(folder,"no_actor_updates",candidate=candidate,target=target,
                training_summary=exc.summary, before_checkpoint=before_checkpoint);continue
        except Exception as exc:
            save(folder,"training_or_audit_failed",candidate=candidate,target=target,
                model_decision="rollback",detail=str(exc),before_checkpoint=before_checkpoint);continue
        checkpoint=proposed if verdict["model_decision"]=="accept" else before_checkpoint
        harness=reduced if verdict["module_decision"]=="retire" else full
        write_json(folder/"retirement.json",verdict)
        save(folder,"internalization_audited",candidate=candidate,target=target,
            model_decision=verdict["model_decision"],module_decision=verdict["module_decision"],
            before_checkpoint=before_checkpoint,proposed_checkpoint=proposed)
    result={**AcceptedAgentState(checkpoint,harness,manifest.fingerprint,protocol,config.cycles,components.sampling_state).to_dict(),"archive":archive}
    write_json(output/"deployment.json",result)
    return result
