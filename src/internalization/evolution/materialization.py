"""Shared host validation, public evidence projection and candidate construction.

Moved unchanged from the single-call API adapter; used by Claude workspace
canonicalization and explicit API test fixtures alike.
"""
from dataclasses import asdict
from pathlib import Path
from .candidate import HarnessCandidate
from ..harness.revision import InternalizationTarget
from ..core.types import write_json

SPEC_PATH = Path(__file__).with_name("PROPOSER_SPEC.md")


def public_execution_trace(trajectory):
    trace={"task_id":trajectory.task_id,
        "steps":[{"step":s.state.step,"context":s.student_prompt,"action":s.action} for s in trajectory.transitions],
        "score":trajectory.success,"total_reward":trajectory.total_reward}
    if hasattr(trajectory,"environment_events"):
        trace["environment_events"]=[asdict(e) for e in trajectory.environment_events]
        trace["public_calls"]=[asdict(c) for c in trajectory.public_calls]
        trace["initial_observation"]=trajectory.initial_observation
    return trace


def validate_evidence(refs,trajectories):
    allowed={(t.task_id,s.state.step) for t in trajectories for s in t.transitions}
    if not isinstance(refs,list) or (allowed and not refs):
        raise ValueError("evidence_refs must cite supplied representative decision steps")
    for ref in refs:
        if (not isinstance(ref,dict) or set(ref)!={"task_id","step"} or
            not isinstance(ref['task_id'],str) or type(ref['step']) is not int or
            (ref['task_id'],ref['step']) not in allowed):
            raise ValueError("Invalid evidence reference: task_id/step absent from representative trajectories")


def search_only_history(history):
    """Require host-created search feedback, never generic lifecycle/status records."""
    result=[]
    for row in history:
        if (not isinstance(row,dict) or set(row)!={"feedback_source","candidate","search_gain","status"} or
            row['feedback_source']!='search' or row['status'] not in ('search_positive','no_search_gain','search_failed')):
            raise ValueError("Proposer history must contain only explicit search feedback")
        gain=row['search_gain']
        allowed={'mean','low','high','paired_gains','rule','significance_test','task_ids','tasks','confidence','bootstrap_samples'}
        if gain is not None and (not isinstance(gain,dict) or set(gain)-allowed):
            raise ValueError("Unknown search feedback gain fields")
        result.append(row)
    return result


def validate_search_input(request):
    allowed=set(request.tasks)
    if any(t.task_id not in allowed for t in request.trajectories) or any(s.task_id not in allowed for s in request.scores):
        raise ValueError("Proposer input outside search tasks")
    return search_only_history(request.history)


def materialize_candidates(store, request, proposals):
    allowed=set(request.tasks)
    results=[];seen=set()
    for i,row in enumerate(proposals):
        try:
            if not isinstance(row,dict) or set(row)!={'patch','rationale','evidence_refs','internalization'}:
                raise ValueError("Candidate must contain exactly rationale, evidence_refs, patch, internalization")
            validate_evidence(row['evidence_refs'],request.trajectories)
            if not isinstance(row['patch'],list) or not row['patch']: raise ValueError("Nonempty patch required")
            patch=store.bind_patch(request.harness,row['patch'])
            for edit in patch:
                if any(task and (task in edit.path or (edit.content is not None and task in edit.content)) for task in allowed):
                    raise ValueError("Patch hard-codes an exact supplied search task ID")
            candidate=HarnessCandidate.create(store,request.harness,patch,row['rationale'],evidence_refs=row['evidence_refs'])
            if candidate.full_revision.version in seen: raise ValueError("duplicate full_revision")
            seen.add(candidate.full_revision.version)
            declaration=row['internalization']
            if declaration is not None:
                try:
                    if (not isinstance(declaration,dict) or set(declaration)!={'target_control_id','removed_behavior'} or
                        not isinstance(declaration['target_control_id'],str)):
                        raise ValueError("Expected at most one named removable-control declaration")
                    target=InternalizationTarget.from_control(store,candidate.full_revision,
                        declaration['target_control_id'],declaration['removed_behavior'])
                    candidate=candidate.with_internalization(target)
                except Exception as exc:
                    write_json(request.output/f'candidate_{i}'/'internalization_declaration_error.json',
                        {'declaration':declaration,'reason':str(exc),'full_revision':candidate.full_revision.version,
                         'fallback':'harness_only','repair_model_calls':0})
            write_json(request.output/f"candidate_{i}.json",candidate.to_dict())
            results.append(candidate)
        except Exception as exc:
            write_json(request.output/f"candidate_{i}_invalid.json",{"proposal":row,"reason":str(exc)})
            results.append({"invalid_proposal":row,"reason":str(exc)})
    return tuple(results)
