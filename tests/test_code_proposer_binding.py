"""One structured API request, host-owned targets, and real sandbox preflight."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from internalization.core.interfaces import Components, ProposalRequest
from internalization.core.manifest import TaskManifest
from internalization.core.trajectory import Trajectory, Transition, RolloutResult
from internalization.core.types import Cost, EpisodeResult, Journal, State, write_json
from internalization.evolution.code_proposer import CodeProposer, CONTRACT
from internalization.evolution.candidate import HarnessCandidate
from internalization.evolution.revision_search import public_history, search_feedback
from internalization.harness.revision import InternalizationTarget, RevisionStore, FileEdit, text_hash
from internalization.outer_loop import LoopConfig, run_outer_loop
from internalization.revision_demo import ROOT, DemoEnvironment, DemoModel, improvement
from internalization.training.rollout import InteractionTaskRunner
from test_named_controls import named_revision


class CodeProposerBindingTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name);self.store=RevisionStore(self.root/'revisions')
        self.parent=self.store.import_directory(ROOT/'examples/versioned_harness/base')
        self.tasks=tuple(f'search-instance-{i:03d}' for i in range(8))
        self.traces=tuple(self.trace(t) for t in self.tasks[:4])

    def trace(self,task):
        state=State(task,task+':0',0,'PUBLIC observed failure')
        return Trajectory(task,state.episode_id,0,'old',self.parent.version,(Transition(state,state.public_history,'fail'),),0.,Cost())

    def request(self,parent=None,name='proposal',count=2):
        return ProposalRequest('old',parent or self.parent,self.tasks,self.traces,
            tuple(EpisodeResult(t,0,0.,Cost()) for t in self.tasks),(),0,count,self.root/name)

    def row(self,text='improve context',*,internalization=None,changes=None):
        return {'rationale':'Repeated failures -> clearer public context -> more appropriate actions',
            'evidence_refs':[{'task_id':self.tasks[0],'step':0}],
            'patch':changes if changes is not None else [{'path':'prompts/system.txt','content':text}],
            'internalization':internalization}

    def proposer(self,rows):
        calls=[]
        def transport(payload):
            calls.append(payload)
            return {'choices':[{'message':{'content':json.dumps({'candidates':rows})}}],
                'usage':{'prompt_tokens':100,'completion_tokens':20}}
        return CodeProposer(self.store,model='mock',transport=transport),calls

    def named_row(self,**kwargs):
        tool=improvement(self.store,self.parent,diagnosis=False).full_revision
        full=named_revision(self.store,tool,**kwargs)
        changes=[{'path':p,'content':s} for p,s in full.files().items() if self.parent.files().get(p)!=s]
        return self.row(changes=changes,internalization={'target_control_id':'review_v1','removed_behavior':'Independent public-context review'})

    def test_one_transport_two_independent_parent_bound_candidates_and_cost(self):
        before=self.parent.files()
        proposer,calls=self.proposer([self.row('first'),self.row('second')])
        candidates=proposer.propose(self.request())
        self.assertEqual(len(calls),1);self.assertEqual(proposer.last_cost.model_calls,1)
        self.assertEqual((proposer.last_cost.input_tokens,proposer.last_cost.output_tokens),(100,20))
        self.assertNotEqual(candidates[0].full_revision.version,candidates[1].full_revision.version)
        for i,c in enumerate(candidates):
            c.validate(self.parent)
            self.assertEqual(c.parent_revision,self.parent.version)
            self.assertEqual(c.patch[0].before_hash,text_hash(before['prompts/system.txt']))
            self.assertIsNone(c.internalization_target)
            self.assertEqual(HarnessCandidate.from_dict(c.to_dict()),c)
        self.assertEqual(self.parent.files(),before)
        public=json.loads(calls[0]['messages'][1]['content'])
        self.assertEqual(len(public['scores']),8);self.assertEqual(len(public['traces']),4)
        self.assertEqual(public['traces'][0]['steps'][0]['step'],0)

    def test_valid_embedded_target_is_constructed_without_second_api_and_keeps_tool(self):
        proposer,calls=self.proposer([self.named_row(),self.row()])
        candidate,_=proposer.propose(self.request());target=candidate.internalization_target
        target.validate_structure();candidate.validate(self.parent)
        self.assertEqual(len(calls),1)
        self.assertEqual(target.target_control_id,'review_v1')
        full,reduced=target.full_revision,target.reduced_revision
        self.assertNotEqual(reduced.version,self.parent.version)
        self.assertEqual([c['enabled'] for c in full.config['controls']],[True,True])
        self.assertEqual([c['enabled'] for c in reduced.config['controls']],[True,False])
        self.assertEqual({p for p in full.files() if full.files()[p]!=reduced.files()[p]},{'config/harness.json'})
        self.assertEqual(InternalizationTarget.from_control(self.store,full,'review_v1',target.removed_behavior).reduced_revision.version,reduced.version)
        restored=HarnessCandidate.from_dict(candidate.to_dict());restored.validate(self.parent)
        checkpoint=self.root/'model';write_json(checkpoint/'mock_model.json',{'trained':True,'broken':False})
        runner=InteractionTaskRunner(DemoEnvironment,DemoModel,max_steps=3)
        result=runner.rollout(str(checkpoint),restored.internalization_target.reduced_revision,(self.tasks[0],),
            seeds=(0,),output=self.root/'rollout')
        self.assertEqual(result.evaluations[0].success,1.)
        self.assertIn('log_query',result.trajectories[0].transitions[0].action)
        self.assertTrue(all('<review_v1>' not in t.student_prompt for t in result.trajectories[0].transitions))
        self.assertIn('<recovery_v1>',result.trajectories[0].transitions[0].student_prompt)

    def test_wrong_reference_task_step_and_scores_only_task_are_invalid(self):
        refs=[{'task_id':'test-private','step':0},{'task_id':self.tasks[0],'step':99},
            {'task_id':self.tasks[7],'step':0},{'task_id':self.tasks[0],'step':False}]
        for i,ref in enumerate(refs):
            row=self.row();row['evidence_refs']=[ref]
            proposer,calls=self.proposer([row,self.row('valid sibling')])
            bad,good=proposer.propose(self.request(name=f'ref-{i}'))
            self.assertIn('evidence reference',bad['reason']);good.validate(self.parent)
            self.assertEqual(len(calls),1)

    def test_malformed_optional_declarations_fail_soft_without_repair_calls(self):
        declarations=[{},[],[{'target_control_id':'review_v1'}],{'target_control_id':'missing','removed_behavior':'review'},
            {'target_control_id':'review_v1','removed_behavior':''},
            {'target_control_id':'review_v1','removed_behavior':'review','teacher_prompt':'not allowed'}]
        for i,declaration in enumerate(declarations):
            row=self.named_row();row['internalization']=declaration
            proposer,calls=self.proposer([row,self.row()]);candidate,_=proposer.propose(self.request(name=f'bad-{i}'))
            candidate.validate(self.parent);self.assertIsNone(candidate.internalization_target)
            self.assertIn('tools/log_query.py',candidate.full_revision.files())
            error=json.loads((self.root/f'bad-{i}/candidate_0/internalization_declaration_error.json').read_text())
            self.assertEqual((error['fallback'],error['repair_model_calls']),('harness_only',0))
            self.assertEqual(len(calls),1)

    def test_disabled_or_dependent_control_gets_no_target(self):
        rows=[self.named_row(composition='sequential_suffix'),self.named_row()]
        for edit in rows[1]['patch']:
            if edit['path']=='config/harness.json':
                cfg=json.loads(edit['content']);cfg['controls'][1]['enabled']=False;edit['content']=json.dumps(cfg)
        proposer,calls=self.proposer(rows)
        candidates=proposer.propose(self.request())
        self.assertEqual(len(calls),1)
        self.assertTrue(all(isinstance(c,HarnessCandidate) and c.internalization_target is None for c in candidates))

    def test_duplicate_second_full_revision_invalid_even_with_different_rationale(self):
        row=self.row();second={**row,'rationale':'other wording'}
        proposer,calls=self.proposer([row,second]);good,bad=proposer.propose(self.request())
        good.validate(self.parent);self.assertIn('duplicate',bad['reason']);self.assertEqual(len(calls),1)

    def test_patch_cannot_memorize_any_search_task_id(self):
        for i,task in enumerate((self.tasks[0],self.tasks[7])):
            proposer,_=self.proposer([self.row('answer for '+task),self.row()])
            bad,good=proposer.propose(self.request(name=f'memorization-{i}'))
            self.assertIn('exact supplied search task ID',bad['reason']);good.validate(self.parent)

    def test_strict_fields_and_invalid_patch_preserve_parent_and_sibling(self):
        invalid=[{**self.row(),'H_minus':{}},{**self.row(),'evidence_refs':[]},
            self.row(changes=[{'path':'../trainer.py','content':'pass'}]),
            self.row(changes=[{'path':'tools/x.py','content':'def broken('}]),
            self.row(changes=[{'path':'tools/x.py','content':'pass','before_hash':'guess'}]),
            self.row(changes=[{'path':'tools/x.py','content':'pass'}]*2)]
        before=self.parent.files()
        for i,row in enumerate(invalid):
            proposer,_=self.proposer([row,self.row()]);bad,good=proposer.propose(self.request(name=f'invalid-{i}'))
            self.assertIsInstance(bad,dict);good.validate(self.parent);self.assertEqual(self.parent.files(),before)

    def test_host_hashes_add_update_delete_and_rejects_other_parent(self):
        files=self.parent.files();files['prompts/obsolete.txt']='old';parent=self.store.snapshot(files)
        changes=[{'path':'prompts/system.txt','content':'new'},{'path':'tools/new.py','content':'pass'},
            {'path':'prompts/obsolete.txt','content':None}]
        proposer,_=self.proposer([self.row(changes=changes),self.row()])
        candidate,_=proposer.propose(self.request(parent));candidate.validate(parent)
        self.assertEqual([p.before_hash for p in candidate.patch],[text_hash(files['prompts/system.txt']),None,text_hash('old')])
        self.assertNotIn('prompts/obsolete.txt',candidate.full_revision.files())
        other=self.store.snapshot({**files,'prompts/system.txt':'other'})
        with self.assertRaisesRegex(ValueError,'base hash mismatch'):self.store.apply(other,candidate.patch)

    def test_parent_tampering_during_request_invalidates_patch(self):
        row=self.row()
        class Tamper(CodeProposer):
            def request_json(_,contract,public,output):
                path=Path(self.parent.path)/'prompts/system.txt';path.chmod(0o644);path.write_text('tampered')
                return {'candidates':[row,row]}
        candidates=Tamper(self.store).propose(self.request())
        self.assertTrue(all('content changed' in c['reason'] for c in candidates))

    def test_input_and_history_do_not_include_dev_retirement_or_final_feedback(self):
        journal=Journal(self.root/'history.jsonl')
        search_feedback(journal,{'id':'previous-search'},{'mean':.2},True)
        journal.append('code_candidate',candidate={'id':'dev-private'},status='eligible',dev_gain={'mean':.9})
        journal.append('candidate_failed',candidate={'id':'private-failure'},reason='retirement-private')
        journal.append('cycle_complete',task='final-test-private',reason='rollback')
        proposer,calls=self.proposer([self.row(),self.row('other')])
        request=replace(self.request(),history=public_history(journal))
        proposer.propose(request)
        public=json.loads(calls[0]['messages'][1]['content'])
        for secret in ['dev-private','retirement-private','final-test-private','eligible','rollback','dev_gain']:
            self.assertNotIn(secret,json.dumps(public))
        for name in ['dev-private','retirement-private','final-test-private']:
            with self.assertRaisesRegex(ValueError,'outside search'):
                proposer.propose(replace(request,trajectories=(self.trace(name),)))
        with self.assertRaisesRegex(ValueError,'explicit search'):
            proposer.propose(replace(request,history=({'dev_gain':.5},)))
        self.assertEqual(len(calls),1)

    def test_old_target_stage_and_backend_field_rejected(self):
        from internalization.command_backend import CommandBackend
        with self.assertRaisesRegex(ValueError,'Unknown backend fields'):
            CommandBackend({'target':['obsolete']},self.root/'ledger')
        result=subprocess.run([sys.executable,'-m','internalization.training.entrypoint','target','--help'],
            cwd=ROOT,capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0);self.assertIn('invalid choice',result.stderr)
        self.assertFalse(hasattr(CodeProposer,'propose_target'))

    def test_real_proposer_subprocess_returns_embedded_snapshot_and_one_call_cost(self):
        from internalization.command_backend import CommandBackend
        request=self.request(name='subprocess')
        write_json(request.output/'mock_response.json',{'candidates':[self.named_row(),self.row()]})
        backend=CommandBackend({'cwd':str(ROOT),'execution':{'mode':'evolution_only','device':'cpu'},
            'propose':['{python}','tests/fixtures/structured_proposer_worker.py',
                       '--request','{request}','--response','{response}'],
            'evaluate':['unused-fixture']},self.root/'process_ledger.jsonl')
        candidate,sibling=backend.components().proposer.propose(request)
        candidate.validate(self.parent);sibling.validate(self.parent)
        self.assertEqual(candidate.internalization_target.target_control_id,'review_v1')
        self.assertFalse(candidate.internalization_target.reduced_revision.config['controls'][1]['enabled'])
        self.assertEqual(len((request.output/'claude_invocations.jsonl').read_text().splitlines()),1)
        response=json.loads((request.output/'response.json').read_text())
        self.assertEqual(response['cost']['model_calls'],1)
        self.assertEqual(response['cost']['input_tokens'],110)
        self.assertEqual(json.loads((request.output/'request.json').read_text())['effective_config'],backend.execution_config)

    def test_duplicate_candidate_not_evaluated_and_dev_result_not_in_search_feedback(self):
        from internalization.evolution.revision_search import search_revisions
        from internalization.evaluation.retirement import RetirementPolicy
        candidate=HarnessCandidate.create(self.store,self.parent,self.store.bind_patch(self.parent,self.row()['patch']),'one')
        duplicate=HarnessCandidate.create(self.store,self.parent,candidate.patch,'different wording')
        class Proposer:
            def propose(_,request):return candidate,duplicate
        calls=[]
        def result(tasks,score):
            return RolloutResult((),tuple(EpisodeResult(t,0,score,Cost()) for t in tasks))
        class Runner:
            def rollout(_,model,harness,tasks,*,seeds,output,training=False):
                calls.append(output)
                return result(tasks,1. if output.name=='search' else 0.)
        journal=Journal(self.root/'duplicate.jsonl')
        selected=search_revisions(Components(Proposer(),Runner(),None),self.request(),
            baseline_search=result(self.tasks,0.),baseline_dev=result(('private-dev',),0.),
            dev_tasks=('private-dev',),seeds=(0,),policy=RetirementPolicy(),journal=journal)
        self.assertFalse(selected.accepted)
        self.assertEqual([p.name for p in calls],['search','dev'])
        feedback=public_history(journal)
        self.assertEqual(feedback[0]['status'],'search_positive')
        self.assertNotIn('private-dev',json.dumps(feedback))
        self.assertIn('duplicate',json.loads((self.root/'candidate_1/rejection.json').read_text())['reason'])

    def loop(self,mode):
        partitions={k:tuple(f'{k}-instance-{i:03d}' for i in range(30)) for k in ('train','search','dev','retirement_0','test')}
        partitions['search']=self.tasks
        manifest=TaskManifest('synthetic','v1',partitions)
        first=self.row() if mode=='null' else self.named_row()
        proposer,calls=self.proposer([first,self.row('sibling')]);seen=[];checks=[];trains=[]
        parent=self.parent
        class Runner:
            def rollout(_,model,harness,tasks,*,seeds,output,training=False):
                seen.append((tasks,output.name,harness.version))
                score=float(harness.version!=parent.version)
                if mode=='reject' and tasks[0].startswith('dev-'):score=0.
                traces=tuple(replace(self.trace(t),success=score,harness_version=harness.version) for t in tasks)
                return RolloutResult(traces,tuple(t.outcome for t in traces))
            def check_internalization(_,model,target,tasks,*,output):
                checks.append(target)
                self.assertEqual(tuple(tasks),self.tasks)
                return {'supported':mode!='preflight_fail','reason':'unsupported fixture'}
        class Trainer:
            def train(*args,**kwargs):trains.append(kwargs);raise AssertionError('A=B cannot train')
        result=run_outer_loop(Components(proposer,Runner(),Trainer()),manifest,'old',self.root/'loop',
            LoopConfig(cycles=1,total_train_steps=1,seeds=(0,)),initial_harness=parent)
        return result,calls,seen,checks,trains

    def test_null_target_accepts_complete_harness_without_preflight_or_training(self):
        result,calls,seen,checks,trains=self.loop('null')
        self.assertEqual(len(calls),1);self.assertEqual((checks,trains),([],[]))
        self.assertEqual(result['archive'][0]['reason'],'accepted_without_internalization')
        self.assertNotEqual(result['harness_revision']['version'],self.parent.version)

    def test_dev_rejection_does_not_execute_preflight_or_train(self):
        result,calls,seen,checks,trains=self.loop('reject')
        self.assertEqual(len(calls),1);self.assertEqual((checks,trains),([],[]))
        self.assertEqual(result['harness_revision']['version'],self.parent.version)

    def test_accepted_embedded_target_preflight_then_contribution_no_second_call(self):
        result,calls,seen,checks,trains=self.loop('target')
        self.assertEqual(len(calls),1);self.assertEqual(len(checks),1);self.assertEqual(trains,[])
        self.assertEqual(result['archive'][0]['reason'],'attribution_failed')
        candidate=HarnessCandidate.from_dict(json.loads((self.root/'loop/cycle_00/proposals/candidate_0.json').read_text()))
        self.assertEqual(checks[0],candidate.internalization_target)
        self.assertEqual({name:version for tasks,name,version in seen if name in ('A','B')},
            {'A':checks[0].full_revision.version,'B':checks[0].reduced_revision.version})
        self.assertFalse((self.root/'loop/cycle_00/target_proposal').exists())

    def test_preflight_failure_keeps_full_candidate_without_training(self):
        result,calls,seen,checks,trains=self.loop('preflight_fail')
        self.assertEqual((len(calls),len(checks),trains),(1,1,[]))
        self.assertEqual(result['archive'][0]['reason'],'accepted_without_internalization')
        self.assertEqual(result['harness_revision']['version'],checks[0].full_revision.version)
        self.assertFalse(any(name in ('A','B','C','D') for _,name,_ in seen))

    def test_declaration_does_not_certify_environment_access(self):
        row=self.named_row(sources={'review_v1':'def run(api,payload):\n    api.environment("observe")\n    return {"suffix":"", "selected":False}\n'})
        proposer,calls=self.proposer([row,self.row()]);candidate,_=proposer.propose(self.request())
        checkpoint=self.root/'model';write_json(checkpoint/'mock_model.json',{'trained':True,'broken':False})
        runner=InteractionTaskRunner(DemoEnvironment,DemoModel,max_steps=3)
        result=runner.check_internalization(str(checkpoint),candidate.internalization_target,self.tasks,
            output=self.root/'preflight')
        self.assertFalse(result['supported']);self.assertEqual(len(calls),1)


if __name__=='__main__':unittest.main()
