"""Real kernel-sandbox execution; synthetic environments/models, never GPU evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from internalization.core.interfaces import Components, ProposalRequest
from internalization.core.types import Journal, write_json
from internalization.evolution.candidate import HarnessCandidate
from internalization.evolution.revision_search import search_revisions, public_history
from internalization.evolution.search import evaluate_tasks
from internalization.evaluation.retirement import RetirementPolicy
from internalization.harness.revision import RevisionStore, FileEdit, HarnessRevision, text_hash
from internalization.harness.sandbox import SandboxedCode, SandboxLimits
from internalization.outer_loop import LoopConfig
from internalization.revision_demo import ROOT, improvement, reduction, run_demo, DemoModel, DemoEnvironment
from internalization.training.rollout import InteractionTaskRunner


class RevisionExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.store=RevisionStore(self.root/'revisions')
        self.parent=self.store.import_directory(ROOT/'examples/versioned_harness/base')
        self.ckpt=self.root/'model'
        write_json(self.ckpt/'mock_model.json',{'trained':False,'broken':False})

    def run_code(self,source,*,sandbox=None):
        files=self.parent.files()
        files['agent/probe.py']=source
        revision=self.store.snapshot(files)
        return (sandbox or SandboxedCode()).call(revision,'agent/probe.py:run',{},lambda *a:None)

    def test_new_registered_tool_really_runs_and_trace_identifies_revision(self):
        full=improvement(self.store,self.parent,diagnosis=False).full_revision
        runner=InteractionTaskRunner(DemoEnvironment,lambda p:DemoModel(p,requires_diagnosis=False),max_steps=3)
        result=runner.rollout(str(self.ckpt),full,('search-0',),seeds=(0,),output=self.root/'rollout')
        trajectory=result.trajectories[0]
        self.assertEqual(trajectory.success,1)
        self.assertEqual(trajectory.harness_version,full.version)
        self.assertIn('log_query',trajectory.transitions[0].action)
        self.assertIn('LOG_QUERY_RESULT',trajectory.transitions[1].student_prompt)
        self.assertEqual(trajectory.cost.tool_calls,2)
        self.assertIn(full.path,(self.root/'rollout/trajectories.jsonl').read_text())
        self.assertIn(full.version,(self.root/'rollout/trajectories.jsonl').read_text())
        self.assertEqual(HarnessRevision.from_dict(full.to_dict()).files(),full.files())
        audit=[json.loads(line) for line in (self.root/'rollout/revision_execution.jsonl').read_text().splitlines()]
        self.assertEqual(sum(r['cost']['model_calls'] for r in audit if r['kind']=='student_generation'),2)
        files=full.files()
        crash=self.store.apply(full,(FileEdit('agent/main.py',text_hash(files['agent/main.py']),
            files['agent/main.py'].replace('action=payload["action"]','raise RuntimeError("dispatch failed")')),))
        with self.assertRaisesRegex(RuntimeError,'dispatch failed'):
            runner.rollout(str(self.ckpt),crash,('search-0',),seeds=(0,),output=self.root/'failed_rollout')
        failed=[json.loads(line) for line in (self.root/'failed_rollout/revision_execution.jsonl').read_text().splitlines()]
        self.assertEqual(sum(r['cost']['model_calls'] for r in failed if r['kind']=='student_generation'),1)
        self.assertEqual(failed[-1]['kind'],'candidate_runtime_failure')

    def test_mixed_reduction_is_not_parent_and_keeps_tool_source_and_registration(self):
        candidate=improvement(self.store,self.parent)
        candidate.validate(self.parent)
        target=reduction(self.store,candidate.full_revision)
        target.validate_structure()
        self.assertNotEqual(target.reduced_revision.version,self.parent.version)
        self.assertEqual(target.full_revision.files()['tools/log_query.py'],target.reduced_revision.files()['tools/log_query.py'])
        self.assertEqual(target.full_revision.files()['config/tools.json'],target.reduced_revision.files()['config/tools.json'])
        self.assertIsNone(target.reduced_revision.config['supervision'])

    def test_patch_denies_protected_paths_syntax_errors_and_wrong_parent_hash(self):
        before=self.parent.files()
        for edit in (FileEdit('../src/internalization/training/trainer.py',None,'pass'),
                     FileEdit('agent/bad.py',None,'def invalid('),
                     FileEdit('config/tools.json','incorrect','{}')):
            with self.subTest(path=edit.path),self.assertRaises((ValueError,SyntaxError)):
                self.store.apply(self.parent,(edit,))
            self.assertEqual(self.parent.files(),before)
        improvement(self.store,self.parent).validate(self.parent)

    def test_landlock_seccomp_block_host_files_network_fork_and_write(self):
        canary=self.root/'secret.txt';canary.write_text('DO NOT READ')
        source=f'''import os, socket
def run(api,payload):
    checks=[]
    for operation in (lambda:open({str(canary)!r}).read(),lambda:open("agent/new.txt","w"),lambda:socket.socket(),lambda:os.fork()):
        try: operation(); checks.append(False)
        except PermissionError: checks.append(True)
    return checks
'''
        self.assertEqual(self.run_code(source),[True]*4)
        self.assertEqual(canary.read_text(),'DO NOT READ')

    def test_missing_kernel_isolation_fails_closed(self):
        # Startup failure never retries with in-process execution.
        with patch('internalization.harness.sandbox.subprocess.Popen',side_effect=OSError('isolation unavailable')):
            with self.assertRaisesRegex(OSError,'isolation unavailable'):
                self.run_code('def run(api,payload): return "must not run"')
        import subprocess
        original=subprocess.Popen
        worker=ROOT/'src/internalization/harness/sandbox_worker.py'
        def fail_bootstrap(argv,**kwargs):
            script=('import runpy\n'
                f'ns=runpy.run_path({str(worker)!r})\n'
                'def unavailable(*args): raise RuntimeError("Landlock ABI unavailable")\n'
                'ns["main"].__globals__["confine"]=unavailable\n'
                'ns["main"]()\n')
            return original([argv[0],'-I','-S','-c',script],**kwargs)
        with patch('internalization.harness.sandbox.subprocess.Popen',side_effect=fail_bootstrap):
            with self.assertRaisesRegex(RuntimeError,'Required candidate isolation unavailable: Landlock ABI unavailable'):
                self.run_code('def run(api,payload): raise AssertionError("candidate must never run")')

    def test_timeout_does_not_mutate_parent_or_prevent_next_candidate(self):
        before=self.parent.files()
        with self.assertRaises(TimeoutError):
            self.run_code('def run(api,payload):\n    while True: pass\n',sandbox=SandboxedCode(SandboxLimits(wall_seconds=.15,cpu_seconds=1)))
        self.assertEqual(self.parent.files(),before)
        self.assertEqual(self.run_code('def run(api,payload): return 7'),7)

    def test_capability_call_budget_cannot_be_changed_by_candidate(self):
        with self.assertRaisesRegex(ValueError,'capability budget'):
            self.run_code('def run(api,payload):\n    api.model("one")\n    api.model("two")\n    return 1\n',
                sandbox=SandboxedCode(SandboxLimits(max_rpc_calls=1)))

    def test_core_history_keeps_prior_observations_across_environment_calls(self):
        from internalization.harness.code_runtime import CodeRuntime
        from internalization.training.rollout import EnvironmentStep
        class Environment:
            def step(self,action):return EnvironmentStep('NEW OBSERVATION',0.,False,0.)
        runtime=CodeRuntime(DemoModel(self.ckpt),self.parent,Environment())
        runtime.prepare('PRIOR OBSERVATION',0)
        outcome=runtime.execute('inspect','PRIOR OBSERVATION',0)
        self.assertIn('PRIOR OBSERVATION',outcome.observation)
        self.assertIn('NEW OBSERVATION',outcome.observation)

    def test_candidate_import_and_global_state_are_fresh_on_every_call(self):
        source='counter=0\ndef run(api,payload):\n    global counter\n    counter+=1\n    return counter\n'
        files=self.parent.files();files['agent/probe.py']=source
        revision=self.store.snapshot(files)
        sandbox=SandboxedCode()
        self.assertEqual([sandbox.call(revision,'agent/probe.py:run',{},lambda *a:None) for _ in range(3)],[1,1,1])
        self.assertEqual(self.run_code('def run(api,payload): return 99'),99)

    def test_teacher_fresh_environment_effect_is_unsupported(self):
        full=improvement(self.store,self.parent,unsupported=True).full_revision
        runner=InteractionTaskRunner(DemoEnvironment,lambda p:DemoModel(p,requires_diagnosis=False),max_steps=3)
        # The same hook is legal during ordinary task execution.
        actual=runner.rollout(str(self.ckpt),full,('search',),seeds=(0,),output=self.root/'normal')
        self.assertEqual(actual.evaluations[0].success,1)
        verdict=runner.check_internalization(str(self.ckpt),reduction(self.store,full),('search',),output=self.root/'check')
        self.assertFalse(verdict['supported'])
        self.assertIn('new environment observation/effect',verdict['reason'])

    def test_search_evaluation_failure_preserves_parent_and_evaluates_sibling(self):
        good=improvement(self.store,self.parent,diagnosis=False)
        bad=HarnessCandidate.create(self.store,self.parent,(FileEdit('agent/crash.py',None,'raise RuntimeError("bad")'),),'invalid evaluation')
        real=InteractionTaskRunner(DemoEnvironment,lambda p:DemoModel(p,requires_diagnosis=False),max_steps=3)
        class Runner:
            def rollout(_,model,harness,*args,**kwargs):
                if harness.version==bad.full_revision.version: raise RuntimeError('evaluation failed')
                return real.rollout(model,harness,*args,**kwargs)
        class Proposer:
            def propose(_,request): return bad,good
        runner=Runner();tasks=('search',);dev=('dev',)
        base=evaluate_tasks(runner,str(self.ckpt),self.parent,tasks,(0,),self.root/'base')
        base_dev=evaluate_tasks(runner,str(self.ckpt),self.parent,dev,(0,),self.root/'base_dev')
        request=ProposalRequest(str(self.ckpt),self.parent,tasks,base.trajectories,base.evaluations,(),0,2,self.root/'proposals')
        journal=Journal(self.root/'events.jsonl')
        selected=search_revisions(Components(Proposer(),runner,None),request,baseline_search=base,baseline_dev=base_dev,
            dev_tasks=dev,seeds=(0,),policy=RetirementPolicy(),journal=journal)
        self.assertEqual(selected.candidate_id,good.candidate_id)
        self.parent.files()
        history=public_history(journal)
        self.assertEqual(history[0]['status'],'failed')
        self.assertFalse(any('dev_gain' in row for row in history))


class RevisionLifecycleTests(unittest.TestCase):
    """Unchanged 30-task/95%/2000-bootstrap gates with mock model transitions."""
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.results={}
        for scenario in ('tool_only','mixed','unsupported','attribution_failed','rollback'):
            cls.results[scenario]=run_demo(cls.root/scenario,scenario,LoopConfig(cycles=1,total_train_steps=1,seeds=(0,)))

    @classmethod
    def tearDownClass(cls): cls.temp.cleanup()

    def test_tool_only_accepts_without_updating_model(self):
        result=self.results['tool_only'];state=result['archive'][0]
        self.assertEqual(state['reason'],'accepted_without_internalization')
        self.assertIn('initial_model',result['checkpoint'])
        self.assertIn('log_query',HarnessRevision.from_dict(result['harness_revision']).files()['config/tools.json'])

    def test_retirement_uses_explicit_reduced_revision_in_all_four_cells(self):
        result=self.results['mixed'];state=result['archive'][0];target=state['target']
        self.assertEqual((state['model_decision'],state['module_decision']),('accept','retire'))
        self.assertEqual(result['harness_revision']['version'],target['reduced_revision']['version'])
        for cell,key in (('A','full_revision'),('B','reduced_revision'),('C','full_revision'),('D','reduced_revision')):
            path=self.root/'mixed/experiment/cycle_00'/cell/'trajectories.jsonl'
            rows=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual({r['trajectory']['harness_version'] for r in rows if r['kind']=='trajectory'},{target[key]['version']})
        self.assertIn('tools/log_query.py',HarnessRevision.from_dict(result['harness_revision']).files())

    def test_incompatible_target_preserves_accepted_full_and_skips_training(self):
        state=self.results['unsupported']['archive'][0]
        self.assertEqual(state['reason'],'accepted_without_internalization')
        self.assertIn('new environment observation/effect',state['detail'])
        self.assertEqual(state['harness_revision']['version'],state['target']['full_revision']['version'])
        self.assertFalse((self.root/'unsupported/experiment/cycle_00/training').exists())

    def test_failed_contribution_gate_keeps_full_without_training(self):
        state=self.results['attribution_failed']['archive'][0]
        self.assertEqual(state['reason'],'attribution_failed')
        self.assertEqual(state['harness_revision']['version'],state['target']['full_revision']['version'])
        self.assertFalse((self.root/'attribution_failed/experiment/cycle_00/training').exists())

    def test_model_rollback_preserves_accepted_full_and_new_tool(self):
        state=self.results['rollback']['archive'][0]
        self.assertEqual((state['model_decision'],state['module_decision']),('rollback','retain'))
        self.assertEqual(state['checkpoint'],state['before_checkpoint'])
        self.assertEqual(state['harness_revision']['version'],state['target']['full_revision']['version'])
        self.assertIn('tools/log_query.py',HarnessRevision.from_dict(state['harness_revision']).files())

    def test_protocol_keeps_thresholds_and_cohorts_disjoint(self):
        manifest=json.loads((self.root/'mixed/manifest.json').read_text())['partitions']
        flattened=[t for tasks in manifest.values() for t in tasks]
        self.assertEqual(len(flattened),len(set(flattened)))
        protocol=json.loads((self.root/'mixed/experiment/protocol.json').read_text())
        self.assertEqual(protocol['retirement']['min_tasks'],30)
        self.assertEqual(protocol['retirement']['bootstrap_samples'],2000)
        self.assertEqual(protocol['retirement']['performance_margin'],.02)


if __name__=='__main__': unittest.main()
