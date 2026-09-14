"""Named-control attribution and composition. Mock policies; real sandbox execution."""
from dataclasses import asdict, replace
import json
import tempfile
import unittest
from pathlib import Path

from internalization.core.interfaces import Components, ProposalRequest
from internalization.core.manifest import TaskManifest
from internalization.core.serialization import trajectory_from_dict
from internalization.core.trajectory import RolloutResult
from internalization.core.types import Cost, EpisodeResult, Journal, write_json
from internalization.evolution.candidate import HarnessCandidate
from internalization.evolution.code_proposer import CodeProposer
from internalization.harness.code_runtime import CodeRuntime
from internalization.harness.control_runtime import score_control_context
from internalization.harness.revision import HarnessRevision, InternalizationTarget, RevisionStore
from internalization.harness.runtime import Completion
from internalization.outer_loop import LoopConfig, run_outer_loop
from internalization.revision_demo import ROOT, DemoEnvironment, DemoModel, improvement
from internalization.training.rollout import InteractionTaskRunner
from internalization.training.teacher_scoring import ModuleTeacherScorer


def named_revision(store,parent,ids=('recovery_v1','review_v1'),*,composition='independent_suffix',sources=None):
    files=parent.files()
    controls=[]
    for cid in ids:
        path=f'controls/{cid}.py'
        files[path]=(sources or {}).get(cid,
            'def run(api,payload):\n    return {"suffix":'+repr(f'\n<{cid}>')+',"selected":True}\n')
        controls.append({'id':cid,'entrypoint':path+':run','enabled':True})
    files['config/harness.json']=json.dumps({'schema':2,'entrypoint':parent.config['entrypoint'],
        'composition':composition,'controls':controls})
    return store.snapshot(files)


class ProbeModel:
    snapshot_id='old-policy'
    def __init__(self): self.calls=[];self.scores=[]
    def prompt_ids(self,prompt): return [ord(c)+1 for c in prompt]
    def generate(self,prompt,*,purpose):
        self.calls.append((purpose,prompt))
        if purpose=='harness_internal':
            return Completion('cached-'+str(len(self.calls)),Cost(3,1,1,1))
        return Completion('finish',Cost(2,2,1),(3,4))
    def score(self,prompt,ids):
        self.scores.append((prompt,tuple(ids)))
        return [-.5 if '<recovery_v1>' in prompt else -1.] * len(ids),Cost(2,0,1)


class NamedControlTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.store=RevisionStore(self.root/'revisions')
        self.parent=self.store.import_directory(ROOT/'examples/versioned_harness/base')
        self.tool_parent=improvement(self.store,self.parent,diagnosis=False).full_revision

    def rollout(self,revision,model=None,name='rollout'):
        model=model or ProbeModel()
        result=InteractionTaskRunner(DemoEnvironment,max_steps=1).rollout(model,revision,('search',),
            seeds=(0,),output=self.root/name,training=True)
        return model,result.trajectories[0]

    def test_named_reduction_disables_exact_id_and_retains_tools_source_and_order(self):
        full=named_revision(self.store,self.tool_parent)
        target=InternalizationTarget.from_control(self.store,full,'review_v1','review only')
        self.assertEqual([c['enabled'] for c in target.reduced_revision.config['controls']],[True,False])
        self.assertEqual([c['id'] for c in target.reduced_revision.config['controls']],['recovery_v1','review_v1'])
        self.assertEqual(target.supervision_adapter,'controls/review_v1.py:run')
        self.assertEqual(InternalizationTarget.from_dict(json.loads(json.dumps(target.to_dict()))),target)
        self.assertEqual({p for p in full.files() if full.files()[p]!=target.reduced_revision.files()[p]}, {'config/harness.json'})
        checkpoint=self.root/'model';write_json(checkpoint/'mock_model.json',{'trained':True,'broken':False})
        runner=InteractionTaskRunner(DemoEnvironment,lambda p:DemoModel(p,requires_diagnosis=False),max_steps=3)
        result=runner.rollout(str(checkpoint),target.reduced_revision,('search',),seeds=(0,),output=self.root/'tools')
        self.assertEqual(result.evaluations[0].success,1)
        prompt=result.trajectories[0].transitions[1].student_prompt
        self.assertIn('LOG_QUERY_RESULT',prompt);self.assertIn('<recovery_v1>',prompt)
        self.assertNotIn('<review_v1>',prompt)

    def test_targets_at_first_middle_last_position_match_full_deployment_context(self):
        full=named_revision(self.store,self.parent,('recovery_v1','review_v1','extra_v1'))
        for i,cid in enumerate(('recovery_v1','review_v1','extra_v1')):
            with self.subTest(target=cid):
                target=InternalizationTarget.from_control(self.store,full,cid,cid)
                model,trajectory=self.rollout(target.reduced_revision,name=f'rollout{i}')
                step=trajectory.transitions[0]
                enhanced,selected,_=score_control_context(model,target,step)
                deployed,_=CodeRuntime(model,full,DemoEnvironment()).prepare(DemoEnvironment().reset('search',0),0)
                self.assertEqual(enhanced,deployed)
                self.assertTrue(selected)
                self.assertNotIn('<'+cid+'>',step.student_prompt)

    def test_non_target_stochastic_outputs_are_cached_and_target_sees_only_base(self):
        sources={
            'recovery_v1':'def run(api,payload):\n    assert "cached-" not in payload["context"]\n    return {"suffix":"\\n<recovery_v1>","selected":True}\n',
            'review_v1':'def run(api,payload):\n    assert "<recovery_v1>" not in payload["context"]\n    return {"suffix":"\\n"+api.model("retained"),"selected":True}\n'}
        full=named_revision(self.store,self.parent,sources=sources)
        target=InternalizationTarget.from_control(self.store,full,'recovery_v1','recover')
        model,trajectory=self.rollout(target.reduced_revision)
        before=list(model.calls)
        scorer=ModuleTeacherScorer(model,full,target,{'search'})
        signals,cost=scorer.score((trajectory,))
        self.assertEqual(model.calls,before)  # No re-generation of the retained stochastic call.
        context=trajectory.transitions[0].control_context
        self.assertEqual(model.scores[-1][0],context.base_context+'\n<recovery_v1>'+context.outputs[0].suffix)
        self.assertTrue(next(iter(signals.values())).selected)
        self.assertEqual(model.scores[-1][1],trajectory.transitions[0].response_ids)
        self.assertEqual(cost.model_calls,1)  # Target is local, only its action-token score costs a call.
        self.assertEqual(trajectory.cost.auxiliary_calls,1)

    def test_serialized_trace_round_trip_and_corrupt_composition_fail_closed(self):
        full=named_revision(self.store,self.parent)
        target=InternalizationTarget.from_control(self.store,full,'recovery_v1','recover')
        model,trajectory=self.rollout(target.reduced_revision)
        restored=trajectory_from_dict(json.loads(json.dumps(asdict(trajectory))))
        self.assertEqual(restored,trajectory)
        step=restored.transitions[0];context=step.control_context
        bad_contexts=[None,replace(context,base_context='hidden observation'),
            replace(context,outputs=()),replace(context,composition='sequential_suffix'),
            replace(context,outputs=(replace(context.outputs[0],control_id='other'),))]
        for bad in bad_contexts:
            with self.subTest(context=bad),self.assertRaises(ValueError):
                score_control_context(model,target,replace(step,control_context=bad))

    def test_named_schema_rejects_duplicate_ids_wrong_flags_and_missing_entrypoints(self):
        full=named_revision(self.store,self.parent)
        invalids=[]
        config=full.config
        invalids.append({**config,'controls':[config['controls'][0]]*2})
        invalids.append({**config,'controls':[{**config['controls'][0],'enabled':1}]})
        invalids.append({**config,'controls':[{**config['controls'][0],'entrypoint':'controls/missing.py:run'}]})
        for bad in invalids:
            files=full.files();files['config/harness.json']=json.dumps(bad)
            with self.assertRaises(ValueError):self.store.snapshot(files)
        for cid in ('missing',''):
            with self.assertRaises(ValueError):InternalizationTarget.from_control(self.store,full,cid,'bad')
        target=InternalizationTarget.from_control(self.store,full,'review_v1','review')
        with self.assertRaises(ValueError):InternalizationTarget.from_control(self.store,target.reduced_revision,'review_v1','disabled')

    def test_structure_rejects_other_control_removal_reordering_or_shared_code_edit(self):
        full=named_revision(self.store,self.parent)
        target=InternalizationTarget.from_control(self.store,full,'review_v1','review')
        for mode in ('disable_other','reorder','source'):
            files=target.reduced_revision.files();config=target.reduced_revision.config
            if mode=='disable_other':config['controls'][0]['enabled']=False
            elif mode=='reorder':config['controls'].reverse()
            else:files['controls/recovery_v1.py']+='\n# changed shared source\n'
            files['config/harness.json']=json.dumps(config)
            reduced=self.store.snapshot(files)
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                replace(target,reduced_revision=reduced).validate_structure()

    def test_dependent_composition_runs_but_is_unsupported_for_internalization(self):
        sources={'review_v1':'def run(api,payload):\n    assert "<recovery_v1>" in payload["context"]\n    return {"suffix":"\\n<review_v1>","selected":True}\n'}
        full=named_revision(self.store,self.parent,composition='sequential_suffix',sources=sources)
        _,trajectory=self.rollout(full)
        self.assertIn('<recovery_v1>\n<review_v1>',trajectory.transitions[0].student_prompt)
        with self.assertRaisesRegex(ValueError,'dependent control composition'):
            InternalizationTarget.from_control(self.store,full,'review_v1','review')
        proposer=CodeProposer(self.store,transport=lambda _:self.fail('Must not call API'))
        request=ProposalRequest('old',full,('search',),(),(),(),0,1,self.root/'target')
        with self.assertRaisesRegex(ValueError,'unsupported'):proposer.propose_target(request)
        self.assertEqual(proposer.last_cost.model_calls,0)

    def test_independent_control_cannot_read_switches_or_request_environment(self):
        for i,source in enumerate(('def run(api,payload):\n    open("config/harness.json").read()\n',
                                  'def run(api,payload):\n    api.environment("observe")\n')):
            full=named_revision(self.store,self.parent,('probe',),sources={'probe':source})
            with self.subTest(source=source),self.assertRaises((RuntimeError,ValueError)):
                self.rollout(full,name=f'forbidden{i}')
        # prepare also cannot derive different base contexts from the ON/OFF config.
        full=named_revision(self.store,self.parent)
        files=full.files();files['agent/main.py']='def run(api,payload): return open("config/harness.json").read()\n'
        with self.assertRaisesRegex(RuntimeError,'PermissionError'):
            self.rollout(self.store.snapshot(files),name='forbidden_prepare')

    def test_named_proposer_selects_only_id_and_host_derives_adapter(self):
        full=named_revision(self.store,self.parent)
        captured=[]
        def transport(payload):
            captured.append(payload)
            return {'choices':[{'message':{'content':json.dumps({'target':{'target_control_id':'review_v1','removed_behavior':'review'}})}}],
                'usage':{'prompt_tokens':10,'completion_tokens':10}}
        proposer=CodeProposer(self.store,model='mock',transport=transport)
        request=ProposalRequest('old',full,('search',),(),(),(),0,1,self.root/'target')
        target=proposer.propose_target(request)
        self.assertEqual(target.target_control_id,'review_v1')
        self.assertEqual(target.supervision_adapter,'controls/review_v1.py:run')
        self.assertEqual(len(captured),1)
        self.assertEqual(json.loads((self.root/'target/internalization_target.json').read_text())['target_control_id'],'review_v1')

    def test_empty_named_registry_skips_api(self):
        full=named_revision(self.store,self.parent,())
        proposer=CodeProposer(self.store,transport=lambda _:self.fail('No controls, no API'))
        request=ProposalRequest('old',full,('search',),(),(),(),0,1,self.root/'target')
        self.assertIsNone(proposer.propose_target(request))
        self.assertEqual(proposer.last_cost.model_calls,0)

    def run_cycles(self,mode='retire'):
        initial=named_revision(self.store,self.tool_parent,(),
            composition='sequential_suffix' if mode=='unsupported' else 'independent_suffix')
        store=self.store;seen=[];trainer_targets=[]
        class Proposer:
            def propose(_,request):
                cid=('recovery_v1','review_v1')[request.cycle]
                config=request.harness.config
                config['controls'].append({'id':cid,'entrypoint':f'controls/{cid}.py:run','enabled':True})
                changes=[{'path':'config/harness.json','content':json.dumps(config)},
                    {'path':f'controls/{cid}.py','content':'def run(api,payload): return {"suffix":'+repr('\n<'+cid+'>')+',"selected":True}\n'}]
                return tuple(HarnessCandidate.create(store,request.harness,store.bind_patch(request.harness,changes),cid+str(i)) for i in range(2))
            def propose_target(_,request):
                cid=request.harness.config['controls'][-1]['id']
                return InternalizationTarget.from_control(store,request.harness,cid,cid)
        class Runner:
            def rollout(_,model,harness,tasks,*,seeds,output,training=False):
                active={c['id'] for c in harness.config['controls'] if c['enabled']}
                seen.append((model,active,tasks,output.name))
                recovery=float('recovery_v1' in active);review=float('review_v1' in active)
                score={'old':.5*recovery+.3*review,'new1':.1+.4*recovery+.3*review,'new2':.6+.4*recovery}[model]
                if mode=='rollback' and model=='new2': score=0.
                return RolloutResult((),tuple(EpisodeResult(t,s,score,Cost(10+10*len(active),2,1+len(active),len(active),1,1.+len(active))) for t in tasks for s in seeds))
            def check_internalization(*a,**kw):return {'supported':True}
        class Trainer:
            def train(_,student,teacher,full,reduced,trajectories,*,target,**kwargs):
                target.validate_structure();trainer_targets.append(target.target_control_id)
                return 'new1' if student=='old' else 'new2'
        partitions={name:tuple(f'{name}-{i}' for i in range(30)) for name in
            ('train','search','dev','test','acceptance_0','acceptance_1','retirement_0','retirement_1')}
        result=run_outer_loop(Components(Proposer(),Runner(),Trainer()),TaskManifest('mock','v1',partitions),'old',
            self.root/'loop',LoopConfig(cycles=2,total_train_steps=2,seeds=(0,)),initial_harness=initial)
        return result,seen,trainer_targets

    def test_two_cycles_retain_recovery_then_retire_review_with_actual_statistical_gates(self):
        result,seen,trainer_targets=self.run_cycles()
        self.assertEqual(trainer_targets,['recovery_v1','review_v1'])
        self.assertEqual([(s['model_decision'],s['module_decision']) for s in result['archive']], [('accept','retain'),('accept','retire')])
        self.assertEqual(result['checkpoint'],'new2')
        final=HarnessRevision.from_dict(result['harness_revision'])
        self.assertEqual([(c['id'],c['enabled']) for c in final.config['controls']],[('recovery_v1',True),('review_v1',False)])
        cells={name:active for model,active,tasks,name in seen if tasks[0].startswith('retirement_1')}
        self.assertEqual(cells,{'A':{'recovery_v1','review_v1'},'B':{'recovery_v1'},'C':{'recovery_v1','review_v1'},'D':{'recovery_v1'}})
        self.assertFalse(any(t.startswith('test') for _,_,tasks,_ in seen for t in tasks))
        _,deployed=self.rollout(final,name='deployed')
        self.assertIn('<recovery_v1>',deployed.transitions[0].student_prompt)
        self.assertNotIn('<review_v1>',deployed.transitions[0].student_prompt)

    def test_named_training_regression_rolls_back_model_and_retains_both_controls(self):
        result,_,_=self.run_cycles('rollback')
        self.assertEqual(result['checkpoint'],'new1')
        self.assertEqual((result['archive'][-1]['model_decision'],result['archive'][-1]['module_decision']),('rollback','retain'))
        full=HarnessRevision.from_dict(result['harness_revision'])
        self.assertTrue(all(c['enabled'] for c in full.config['controls']))
        self.assertIn('tools/log_query.py',full.files())

    def test_unsupported_composition_preserves_accepted_revision_without_training(self):
        result,_,trained=self.run_cycles('unsupported')
        self.assertEqual(trained,[])
        self.assertEqual(result['checkpoint'],'old')
        for state in result['archive']:
            self.assertEqual(state['reason'],'accepted_without_internalization')
            self.assertIn('unsupported',state['detail'])
        full=HarnessRevision.from_dict(result['harness_revision'])
        self.assertEqual(len(full.config['controls']),2)
        self.assertTrue(all(c['enabled'] for c in full.config['controls']))


if __name__=='__main__': unittest.main()
