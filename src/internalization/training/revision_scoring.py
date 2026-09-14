"""Executable target bridge; same behavior policy, actual student prompt and response IDs."""
from dataclasses import asdict
import math

from ..core.types import Cost, Journal, write_json
from ..harness.code_runtime import augment_context
from ..harness.control_runtime import score_control_context
from .teacher_scoring import ModuleSignal


def score_revision(scorer,trajectories):
    target=scorer.target
    target.validate_structure()
    if target.full_revision.version != scorer.harness.version: raise ValueError("Target/full revision mismatch")
    policy=scorer.teacher
    if policy.snapshot_id != scorer.snapshot_id: raise RuntimeError("Behavior policy changed")
    if hasattr(policy,"assert_frozen"): policy.assert_frozen()
    signals,cost={},Cost()
    for trajectory in trajectories:
        if trajectory.model_version != scorer.snapshot_id or trajectory.harness_version != target.reduced_revision.version:
            raise ValueError("Revision teacher requires current behavior-policy H-minus rollouts")
        if trajectory.task_id not in scorer.allowed_tasks: raise ValueError("Teacher task outside allowlist")
        for step in trajectory.transitions:
            if step.state.public_history != step.student_prompt:
                raise ValueError("Revision teacher can see only the actual student context")
            if tuple(policy.prompt_ids(step.student_prompt)) != step.prompt_ids: raise ValueError("Prompt tokenization mismatch")
            if len(step.response_ids)!=len(step.old_log_probs) or not step.response_ids:
                raise ValueError("Missing current rollout response IDs/log probabilities")
            if step.state.fingerprint in signals: raise ValueError("Duplicate student state")
            if target.target_control_id is not None:
                enhanced,selected,used=score_control_context(policy,target,step,audit=scorer.journal)
            else:
                enhanced,selected,used=augment_context(policy,target.full_revision,step.student_prompt,step.state.step,audit=scorer.journal)
            cost+=used
            selected=selected or scorer.mode=="all"
            lp=()
            if selected:
                lp,used=policy.score(enhanced,list(step.response_ids));cost+=used
                if len(lp)!=len(step.response_ids) or any(not math.isfinite(x) or x>1e-5 for x in lp):
                    raise ValueError("Invalid teacher action-token scores")
            signal=ModuleSignal(selected,tuple(lp),scorer.snapshot_id,step.state.fingerprint)
            signals[step.state.fingerprint]=signal
            if scorer.journal: scorer.journal.append("revision_teacher_state",state=asdict(step.state),
                student_prompt=step.student_prompt,enhanced_context=enhanced,response_ids=step.response_ids,
                signal=asdict(signal),full_revision=target.full_revision.version,reduced_revision=target.reduced_revision.version)
    if hasattr(policy,"assert_frozen"): policy.assert_frozen()
    if policy.snapshot_id != scorer.snapshot_id: raise RuntimeError("Behavior policy changed")
    return signals,cost


def check_internalization(runner,model,target,tasks,*,output):
    from .behavior_policy import BehaviorPolicySnapshot
    from .teacher_scoring import ModuleTeacherScorer
    output.mkdir(parents=True,exist_ok=True)
    cost=Cost()
    try:
        target.validate_structure()
        if isinstance(model,str): model=runner.model_loader(model)
        # Real student states, using only search tasks. Never retirement/test trajectories.
        with BehaviorPolicySnapshot(model) as behavior:
            result=runner.rollout(behavior,target.reduced_revision,tasks[:1],seeds=(0,),output=output/"rollout",training=True)
            cost=sum((t.cost for t in result.trajectories),Cost())
            if not any(t.transitions for t in result.trajectories): raise ValueError("unsupported: no student action states")
            scorer=ModuleTeacherScorer(behavior,target.full_revision,target,tasks,journal=Journal(output/"scoring.jsonl"))
            _,used=scorer.score(result.trajectories);cost+=used
        verdict={"supported":True,"reason":None,"cost":asdict(cost)}
    except Exception as exc:
        verdict={"supported":False,"reason":str(exc),"cost":asdict(cost)}
    write_json(output/"compatibility.json",verdict)
    return verdict
