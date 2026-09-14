"""Executable named-control/replay demonstration; scripted model, no optimizer updates."""
import argparse
from pathlib import Path

from internalization.core.types import Journal, write_json
from internalization.harness.revision import RevisionStore, InternalizationTarget
from internalization.revision_demo import ROOT, DemoEnvironment, DemoModel
from internalization.training.behavior_policy import BehaviorPolicySnapshot
from internalization.training.rollout import InteractionTaskRunner
from internalization.training.teacher_scoring import ModuleTeacherScorer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,default=ROOT/'examples/named_controls')
    parser.add_argument('--target-control-id',default='review_v1')
    args=parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    store=RevisionStore(output/'revisions')
    full=store.import_directory(args.workspace)
    target=InternalizationTarget.from_control(store,full,args.target_control_id,'Bypass only '+args.target_control_id)
    write_json(output/'internalization_target.json',target.to_dict())
    checkpoint=output/'scripted_model'
    write_json(checkpoint/'mock_model.json',{'trained':True,'broken':False})
    # "trained" is a scripted fixture switch, not a learned checkpoint or training result.
    model=DemoModel(checkpoint,requires_diagnosis=False)
    runner=InteractionTaskRunner(DemoEnvironment,max_steps=3)
    with BehaviorPolicySnapshot(model) as policy:
        minus=runner.rollout(policy,target.reduced_revision,('synthetic-public-log',),seeds=(0,),
            output=output/'student',training=True)
        signals,cost=ModuleTeacherScorer(policy,full,target,{'synthetic-public-log'},
            journal=Journal(output/'scoring.jsonl')).score(minus.trajectories)
        plus=runner.rollout(policy,full,('synthetic-public-log',),seeds=(0,),output=output/'full')
    active=lambda rev:[c['id'] for c in rev.config['controls'] if c['enabled']]
    write_json(output/'proof.json',{'mode':'scripted CPU model; actual sandbox and same-state scoring',
        'full_revision':full.to_dict(),'reduced_revision':target.reduced_revision.to_dict(),
        'target_control_id':target.target_control_id,'full_enabled':active(full),'reduced_enabled':active(target.reduced_revision),
        'student_success':minus.evaluations[0].success,'full_success':plus.evaluations[0].success,
        'student_tool_calls':minus.trajectories[0].cost.tool_calls,
        'selected_states':sum(s.selected for s in signals.values()),'scoring_model_calls':cost.model_calls,
        'actual_optimizer_updates':0,'real_model':False,'GPU':False,'official_benchmark':False})
    print(output/'proof.json')


if __name__=='__main__': main()
