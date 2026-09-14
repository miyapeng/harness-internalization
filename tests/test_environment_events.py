"""Environment returns survive tool execution without inventing policy decisions."""
from dataclasses import asdict, replace
import json
import tempfile
import unittest
from pathlib import Path

import test_behavior_policy as fixtures
from internalization.core.interfaces import ProposalRequest
from internalization.core.serialization import trajectory_from_dict
from internalization.core.trajectory import EventTrajectory, Trajectory, read_trace_file
from internalization.core.types import Cost
from internalization.evolution.code_proposer import CodeProposer
from internalization.harness.revision import RevisionStore, InternalizationTarget
from internalization.harness.runtime import Completion
from internalization.revision_demo import ROOT
from internalization.training.module_advantage import AdvantageConfig
from internalization.training.rollout import EnvironmentStep, InteractionTaskRunner
from internalization.training.teacher_scoring import ModuleSignal
from internalization.training.trainer import ModuleTrainer, build_update_batch


class PublicEnvironment:
    hidden_answer='DO_NOT_SERIALIZE_HIDDEN_REFERENCE'
    def reset(self,task,seed): return 'START '+task
    def step(self,action):
        reward,done={'probe':(.25,False),'partial':(.5,False),'finish':(3.,True),'act':(2.,True)}[action]
        return EnvironmentStep('PUBLIC '+action,reward,done,float(done))
    def close(self): pass


class EventPolicy:
    snapshot_id='old-policy'
    pad_token_id=0
    def __init__(self): self.generated=[];self.scored=[]
    def prompt_ids(self,prompt): return [1,2]
    def generate(self,prompt,*,purpose):
        self.generated.append((prompt,purpose))
        if purpose=='harness_internal':return Completion('PUBLIC GUIDANCE',Cost(2,1,1,1))
        return Completion('act',Cost(2,2,1),(3,4))
    def score(self,prompt,ids):
        self.scored.append((prompt,tuple(ids)))
        return [-1.]*len(ids),Cost(4,0,1)


class EnvironmentEventTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.store=RevisionStore(self.root/'revisions')
        self.base=self.store.import_directory(ROOT/'examples/versioned_harness/base')

    def revision(self,prepare='',execute='api.environment("finish")',hook=None):
        files=self.base.files()
        source='def run(api,payload):\n    memory=payload["memory"]\n    if payload["operation"]=="prepare":\n'
        if prepare: source+='\n'.join('        '+line for line in prepare.splitlines())+'\n'
        source+='        return {"prompt":payload["history"],"tools":{},"memory":memory}\n'
        source+='\n'.join('    '+line for line in execute.splitlines())+'\n'
        source+='    return {"observation":"PUBLIC LOCAL RESULT","memory":memory,"stop":False}\n'
        files['agent/main.py']=source
        if hook:
            files['controls/probe.py']=hook
            files['config/harness.json']=json.dumps({**self.base.config,'supervision':'controls/probe.py:run'})
        return self.store.snapshot(files)

    def run_revision(self,revision,*,max_steps=3,name='rollout',policy=None):
        policy=policy or EventPolicy()
        result=InteractionTaskRunner(PublicEnvironment,max_steps=max_steps).rollout(policy,revision,('search',),
            seeds=(0,),output=self.root/name,training=True)
        return policy,result.trajectories[0]

    def batch(self,trajectory,policy):
        signals={s.state.fingerprint:ModuleSignal(False,(),trajectory.model_version,s.state.fingerprint) for s in trajectory.transitions}
        return build_update_batch((trajectory,),signals,policy,AdvantageConfig())

    def test_prepare_terminal_reward_without_student_or_auxiliary_generation(self):
        revision=self.revision('api.environment("finish")',hook='def run(api,payload): raise RuntimeError("must not run after done")\n')
        policy,trajectory=self.run_revision(revision)
        self.assertEqual(trajectory.success,1.)
        self.assertEqual(trajectory.total_reward,3.)
        self.assertEqual(trajectory.transitions,())
        self.assertEqual(policy.generated,[]);self.assertEqual(policy.scored,[])
        self.assertEqual(trajectory.cost.tool_calls,1)
        self.assertEqual(trajectory.cost.model_calls,0)
        event,=trajectory.environment_events
        self.assertEqual((event.phase,event.step,event.reward,event.done),('prepare',0,3.,True))
        with self.assertRaisesRegex(ValueError,'No student states'):self.batch(trajectory,policy)

    def test_terminal_next_prepare_credits_previous_real_decision_without_extra_tokens(self):
        revision=self.revision('if memory.get("acted"):\n    api.environment("finish")',
            'api.environment("partial")\nmemory["acted"]=True')
        policy,trajectory=self.run_revision(revision)
        self.assertEqual(len(trajectory.transitions),1)
        self.assertEqual(trajectory.transitions[0].reward,.5)
        self.assertEqual(trajectory.total_reward,3.5)
        self.assertEqual([(e.phase,e.step) for e in trajectory.environment_events],[('execute',0),('prepare',1)])
        batch=self.batch(trajectory,policy).tensor_data
        self.assertEqual(batch['responses'].tolist(),[[3,4]])
        self.assertEqual(batch['token_level_rewards'].tolist(),[[0.,3.5]])
        self.assertEqual(batch['response_mask'].tolist(),[[1,1]])

    def test_multiple_environment_calls_count_once_across_prepare_and_execute(self):
        revision=self.revision('api.environment("probe")\napi.environment("partial")')
        policy,trajectory=self.run_revision(revision)
        self.assertEqual([e.reward for e in trajectory.environment_events],[.25,.5,3.])
        self.assertEqual(trajectory.total_reward,3.75)
        self.assertEqual(trajectory.transitions[0].reward,3.75)
        self.assertEqual(self.batch(trajectory,policy).tensor_data['token_level_rewards'].tolist(),[[0.,3.75]])
        self.assertEqual(trajectory.cost.tool_calls,3)

    def test_prepare_environment_and_execute_local_tool_both_have_records_and_costs(self):
        revision=self.revision('api.environment("probe")','memory["local"]=True')
        _,trajectory=self.run_revision(revision,max_steps=1)
        self.assertEqual(trajectory.total_reward,.25)
        self.assertEqual(trajectory.cost.tool_calls,2)
        self.assertEqual([c.operation for c in trajectory.public_calls],['environment','local_tool'])
        self.assertEqual(trajectory.public_calls[-1].parameters,{'action':'act'})
        self.assertEqual(trajectory.public_calls[-1].result['observation'],'PUBLIC LOCAL RESULT')

    def test_empty_event_ledger_is_authoritative_and_old_trajectory_still_loads(self):
        _,trajectory=self.run_revision(self.revision(execute='pass'),max_steps=1)
        self.assertEqual(trajectory.environment_events,())
        tampered=replace(trajectory,transitions=(replace(trajectory.transitions[0],reward=99.),))
        self.assertEqual(tampered.total_reward,0.)
        new=trajectory_from_dict(json.loads(json.dumps(asdict(tampered))))
        self.assertIsInstance(new,EventTrajectory);self.assertEqual(new.total_reward,0.)
        old=asdict(tampered);old.pop('environment_events');old.pop('public_calls')
        old.pop('initial_observation')
        restored=trajectory_from_dict(old)
        self.assertIs(type(restored),Trajectory);self.assertEqual(restored.total_reward,99.)

    def test_event_and_public_call_serialization_round_trip_including_no_response(self):
        _,trajectory=self.run_revision(self.revision('api.environment("finish")'))
        self.assertEqual(trajectory_from_dict(json.loads(json.dumps(asdict(trajectory)))),trajectory)
        self.assertEqual(read_trace_file(self.root/'rollout/trajectories.jsonl'),(trajectory,))

    def test_call_audit_records_public_parameters_results_and_terminal_event_before_failure(self):
        revision=self.revision('api.model("PUBLIC MODEL PROMPT")\napi.environment("finish")\nraise RuntimeError("after tool")')
        with self.assertRaisesRegex(RuntimeError,'after tool'):self.run_revision(revision)
        rows=[json.loads(line) for line in (self.root/'rollout/revision_execution.jsonl').read_text().splitlines()]
        calls=[row for row in rows if row['kind']=='capability']
        self.assertEqual(calls[0]['parameters'],{'prompt':'PUBLIC MODEL PROMPT'})
        self.assertEqual(calls[0]['result'],'PUBLIC GUIDANCE')
        self.assertEqual(calls[1]['parameters'],{'action':'finish'})
        self.assertEqual(calls[1]['result'],{'observation':'PUBLIC finish','done':True,'action_valid':True})
        event=next(row for row in rows if row['kind']=='environment_event')
        self.assertEqual((event['task_id'],event['phase'],event['reward']),('search','prepare',3.))
        self.assertEqual(rows[-1]['kind'],'candidate_runtime_failure')
        self.assertNotIn(PublicEnvironment.hidden_answer,json.dumps(rows))

    def test_proposer_sees_prepare_operations_even_when_there_are_no_student_actions(self):
        _,trajectory=self.run_revision(self.revision('api.model("PUBLIC PROMPT")\napi.environment("finish")'))
        captured=[]
        def transport(payload):
            captured.append(payload)
            return {'choices':[{'message':{'content':json.dumps({'candidates':[{'patch':[],'rationale':'inspect public evidence'}]})}}],
                'usage':{'prompt_tokens':10,'completion_tokens':10}}
        proposer=CodeProposer(self.store,model='mock',transport=transport)
        request=ProposalRequest('old',self.base,('search',),(trajectory,),(trajectory.outcome,),(),0,1,self.root/'proposal')
        proposer.propose(request)
        trace=json.loads(captured[0]['messages'][1]['content'])['traces'][0]
        self.assertEqual(trace['steps'],[])
        self.assertEqual(trace['total_reward'],3.)
        self.assertEqual(trace['initial_observation'],'START search')
        self.assertEqual(trace['environment_events'][0]['action'],'finish')
        self.assertEqual([c['operation'] for c in trace['public_calls']],['model','environment'])
        self.assertNotIn(PublicEnvironment.hidden_answer,json.dumps(captured))
        with self.assertRaisesRegex(ValueError,'outside search'):
            proposer.propose(replace(request,tasks=('different-search-task',),output=self.root/'bad-proposal'))

    def training(self,tasks,budget=1,tasks_per_batch=1):
        policy=fixtures.BehaviorPolicyTests().policy()
        reference=fixtures.BehaviorPolicyTests().reference()
        full=self.revision('if "auto" in payload["history"]:\n    api.environment("finish")',
            'api.environment(payload["action"])',
            hook='def run(api,payload): return {"suffix":"","selected":False}\n')
        target=InternalizationTarget.from_supervision(self.store,full,'unused guidance',full.config['supervision'])
        trainer=ModuleTrainer(InteractionTaskRunner(PublicEnvironment,max_steps=1),
            tasks_per_batch=tasks_per_batch,rollouts_per_task=1)
        checkpoint=trainer.train(policy,reference,full,target.reduced_revision,target=target,
            tasks=tasks,budget=budget,output=self.root/'training')
        return policy,trainer,json.loads((self.root/'training/training_summary.json').read_text()),checkpoint

    def test_all_tool_only_batches_skip_actor_update_and_keep_returns_costs(self):
        policy,trainer,summary,checkpoint=self.training(('auto',),budget=2)
        self.assertEqual(policy.updates,0);self.assertEqual(policy.batches,[])
        self.assertEqual(policy.events,[])
        self.assertEqual(policy.snapshot_id,'policy-0')
        self.assertEqual(summary['actor_update_calls'],0)
        self.assertEqual(summary['batches_skipped_no_student_decisions'],2)
        self.assertEqual(trainer.last_cost.tool_calls,2)
        self.assertEqual(summary['cost']['model_calls'],0)
        rows=[json.loads(line) for line in (self.root/'training/training.jsonl').read_text().splitlines()]
        self.assertEqual([r['reason'] for r in rows],['no_student_decisions']*2)
        self.assertEqual([r['episodes'][0]['total_reward'] for r in rows],[3.,3.])
        self.assertEqual(json.loads((Path(checkpoint)/'internalization.json').read_text())['step'],0)

    def test_mixed_batch_only_real_decision_gets_its_own_episode_return(self):
        policy,_,summary,_=self.training(('auto','normal'),tasks_per_batch=2)
        self.assertEqual(policy.updates,1)
        batch=policy.batches[0]
        self.assertEqual(batch['responses'].tolist(),[[3,4]])
        self.assertEqual(batch['token_level_rewards'].tolist(),[[0.,2.]])
        self.assertTrue(batch['response_mask'].all())
        self.assertEqual(summary['batches_skipped_no_student_decisions'],0)
        traces=read_trace_file(self.root/'training/rollout_00000/trajectories.jsonl')
        self.assertEqual([(t.task_id,t.total_reward,len(t.transitions)) for t in traces],[('auto',3.,0),('normal',2.,1)])

    def test_skipped_batch_does_not_drift_policy_before_following_real_batch(self):
        policy,_,summary,checkpoint=self.training(('auto','normal'),budget=2)
        self.assertEqual(summary['actor_update_calls'],1)
        self.assertEqual(summary['batches_skipped_no_student_decisions'],1)
        self.assertEqual({e[1] for e in policy.events if e[0]=='score'},{'policy-0'})
        self.assertEqual(policy.snapshot_id,'policy-1')
        self.assertEqual(json.loads((Path(checkpoint)/'internalization.json').read_text())['step'],1)


if __name__=='__main__': unittest.main()
