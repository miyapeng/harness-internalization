"""State/manifest wiring: mock search scores, actual isolated candidate execution at final evaluation."""
from dataclasses import asdict, replace
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from internalization.benchmarks.importers import make_catalog_manifest
from internalization.benchmarks.appworld_manifest import build_manifest
from internalization.core.accepted_state import AcceptedAgentState, load_accepted_state
from internalization.core.interfaces import Components
from internalization.core.manifest import TaskManifest
from internalization.core.trajectory import RolloutResult
from internalization.core.types import Cost, EpisodeResult, write_json
from internalization.evaluation.attribution import AttributionPolicy
from internalization.evaluation.retirement import RetirementPolicy
from internalization.harness.module import Harness, HarnessModule
from internalization.harness.revision import RevisionStore
from internalization.outer_loop import LoopConfig, run_outer_loop
from internalization.revision_demo import ROOT, DemoEnvironment, DemoModel, DemoProposer, improvement
from internalization.training.rollout import InteractionTaskRunner


def records(count,split='train'):
    return [{'id':f'{split}-{i:04d}','split':split,'record':{'question':f'{split} question {i}'}} for i in range(count)]


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


class AcceptedStatePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.store=RevisionStore(self.root/'revisions')
        self.parent=self.store.import_directory(ROOT/'examples/versioned_harness/base')
        self.full=improvement(self.store,self.parent,diagnosis=False).full_revision
        self.checkpoint=self.root/'model';write_json(self.checkpoint/'mock_model.json',{'trained':False,'broken':False})
        _,self.manifest=make_catalog_manifest('hotpotqa','fixture-v1',records(1,'test'),records(150),cycles=1)
        self.config=LoopConfig(cycles=1,total_train_steps=1,seeds=(0,))

    def agent(self,*,harness=None,manifest=None,config=None,next_cycle=1):
        manifest=manifest or self.manifest;config=config or self.config
        protocol={'mode':'versioned_code_selective_internalization','loop':asdict(config),
            'retirement':asdict(RetirementPolicy()),'attribution':asdict(AttributionPolicy()),
            'harness_acceptance':asdict(AttributionPolicy()),'manifest_hash':manifest.fingerprint}
        return AcceptedAgentState(str(self.checkpoint),harness or self.full,manifest.fingerprint,protocol,next_cycle)

    def state(self,value=None):
        path=self.root/'state.json';write_json(path,value or self.agent().to_dict());return path

    def test_import_allocates_actual_cycle_count_without_borrowing_test(self):
        for cycles in (1,2,4):
            _,manifest=make_catalog_manifest('hotpotqa','v1',records(2,'test'),records((3+cycles)*30),cycles=cycles)
            manifest.validate_loop(cycles,versioned=True,cohort_minimum=30)
            self.assertEqual(len(manifest.partitions['train']),30)
            self.assertEqual(sum(k.startswith('acceptance_') for k in manifest.partitions),0)
            self.assertEqual(manifest.partitions['test'],('test-0000','test-0001'))
        with self.assertRaisesRegex(ValueError,'at least 180'):
            make_catalog_manifest('hotpotqa','v1',records(1,'test'),records(179))
        _,legacy=make_catalog_manifest('hotpotqa','v1',records(1,'test'),records(180),versioned=False)
        self.assertFalse(any(k.startswith('acceptance_') for k in legacy.partitions))

    def test_appworld_allocates_disjoint_scenario_families_without_acceptance(self):
        splits={};offset=0
        for name,count in (('train',80),('dev',30),('test_normal',2),('test_challenge',2)):
            splits[name]=[f'{i:07x}_{j}' for i in range(offset,offset+count) for j in (1,2,3)];offset+=count
        manifest=build_manifest(splits,'v1',cycles=2)
        manifest.validate_loop(2,versioned=True,cohort_minimum=30)
        families={}
        for partition,ids in manifest.partitions.items():
            for task in ids:self.assertEqual(families.setdefault(task.rsplit('_',1)[0],partition),partition)

    def test_preflight_reports_all_missing_cohorts_before_any_backend_call(self):
        manifest=replace(self.manifest,partitions={k:v for k,v in self.manifest.partitions.items() if not k.startswith(('acceptance','retirement'))})
        class Never:
            def propose(*a,**kw):raise AssertionError('proposer ran')
            def rollout(*a,**kw):raise AssertionError('runner ran')
        with self.assertRaisesRegex(ValueError,'retirement_0'):
            run_outer_loop(Components(Never(),Never(),None),manifest,str(self.checkpoint),self.root/'run',self.config,initial_harness=self.parent)
        self.assertFalse((self.root/'run').exists())

    def test_canonical_state_roundtrip_checks_checkpoint_revision_and_identity(self):
        path=self.state();agent=load_accepted_state(path,self.manifest)
        self.assertEqual(agent.harness.version,self.full.version)
        self.assertEqual(agent.checkpoint,str(self.checkpoint))
        self.assertEqual(json.dumps(agent.to_dict(),sort_keys=True),json.dumps(self.agent().to_dict(),sort_keys=True))
        wrong=self.root/'other-model';wrong.mkdir()
        with self.assertRaisesRegex(ValueError,'checkpoint mismatch'):load_accepted_state(path,self.manifest,checkpoint=wrong)
        with self.assertRaisesRegex(ValueError,'manifest identity mismatch'):
            load_accepted_state(path,replace(self.manifest,environment_revision='different'))

    def test_corrupted_protocol_and_revision_never_fall_back(self):
        value=self.agent().to_dict();value['protocol_hash']='wrong'
        with self.assertRaisesRegex(ValueError,'protocol hash'):load_accepted_state(self.state(value),self.manifest)
        path=self.root/'bad-revision.json';value=self.agent().to_dict();value['harness_revision']['version']='wrong'
        write_json(path,value)
        with self.assertRaisesRegex(ValueError,'content changed'):load_accepted_state(path,self.manifest)

    def test_missing_identity_and_unknown_state_are_rejected(self):
        value={'checkpoint':str(self.checkpoint),'harness_revision':self.full.to_dict()}
        with self.assertRaisesRegex(ValueError,'protocol and manifest'):load_accepted_state(self.state(value),self.manifest)
        path=self.root/'unknown.json';write_json(path,{'checkpoint':str(self.checkpoint),'active_modules':[]})
        with self.assertRaisesRegex(ValueError,'no empty-Harness fallback'):load_accepted_state(path,self.manifest)

    def test_historical_code_state_uses_sidecar_and_old_module_state_keeps_source(self):
        cycle=self.root/'old/cycle_00';cycle.mkdir(parents=True)
        write_json(cycle.parent/'protocol.json',self.agent().protocol)
        write_json(cycle/'state.json',{'checkpoint':str(self.checkpoint),'harness_revision':self.full.to_dict(),'cycle':0})
        self.assertEqual(load_accepted_state(cycle/'state.json',self.manifest).next_cycle,1)
        module=HarnessModule.from_source((ROOT/'tests/fixtures/historical_recovery.txt').read_text())
        old=Harness((module,))
        path=cycle/'legacy.json';write_json(path,{'checkpoint':str(self.checkpoint),'harness_version':old.version,
            'active_modules':[{'source':module.source}]})
        self.assertEqual(load_accepted_state(path,self.manifest).harness.version,old.version)
        path=cycle/'names.json';write_json(path,{'checkpoint':str(self.checkpoint),'harness_version':old.version,'active_modules':[module.name]})
        with self.assertRaisesRegex(ValueError,'names only'):load_accepted_state(path,self.manifest)

    def test_completed_or_changed_protocol_cannot_reuse_cohorts(self):
        agent=self.agent()
        with self.assertRaisesRegex(ValueError,'No unconsumed cycles'):agent.check_resume(self.manifest,self.config,RetirementPolicy(),AttributionPolicy())
        incomplete=replace(agent,next_cycle=0)
        with self.assertRaisesRegex(ValueError,'protocol mismatch: loop'):
            incomplete.check_resume(self.manifest,replace(self.config,total_train_steps=2),RetirementPolicy(),AttributionPolicy())
        with self.assertRaisesRegex(ValueError,'completed cycle state'):
            replace(agent,next_cycle=None).check_resume(self.manifest,self.config,RetirementPolicy(),AttributionPolicy())

    def test_historical_module_resume_preserves_recorded_attribution_policy(self):
        _,manifest=make_catalog_manifest('hotpotqa','v1',records(1,'test'),records(180),versioned=False)
        config=LoopConfig(total_train_steps=3,seeds=(0,))
        cycle=self.root/'legacy/cycle_00';cycle.mkdir(parents=True)
        protocol={'loop':asdict(config),'retirement':asdict(RetirementPolicy()),'manifest_hash':manifest.fingerprint}
        write_json(cycle.parent/'protocol.json',protocol)
        policy=AttributionPolicy(min_external_gain=.1)
        write_json(cycle.parent/'attribution_policy.json',asdict(policy))
        write_json(cycle/'state.json',{'checkpoint':str(self.checkpoint),'harness_version':Harness().version,'active_modules':[]})
        agent=load_accepted_state(cycle/'state.json',manifest)
        agent.check_resume(manifest,config,RetirementPolicy(),policy)
        with self.assertRaisesRegex(ValueError,'protocol mismatch: attribution'):
            agent.check_resume(manifest,config,RetirementPolicy(),AttributionPolicy())

    def components(self,proposer,seen):
        class Runner:
            def rollout(_,model,harness,tasks,*,seeds,output,training=False):
                seen.append((model,harness.version,tasks))
                success=float('tools/log_query.py' in harness.files())
                return RolloutResult((),tuple(EpisodeResult(t,s,success,Cost(10,1,1)) for t in tasks for s in seeds))
        class Trainer:
            def train(*a,**kw):raise AssertionError('No-target candidate must not train')
        return Components(proposer,Runner(),Trainer())

    def test_resume_uses_accepted_pair_and_only_unconsumed_cohorts(self):
        _,manifest=make_catalog_manifest('hotpotqa','v1',records(1,'test'),records(210),cycles=2)
        config=LoopConfig(cycles=2,total_train_steps=2,seeds=(0,))
        class InterruptAfterFirst(DemoProposer):
            def propose(self,request):
                if request.cycle==1:raise KeyboardInterrupt('simulated interruption between cycles')
                return super().propose(request)
        first=self.root/'first';seen=[]
        with self.assertRaises(KeyboardInterrupt):
            run_outer_loop(self.components(InterruptAfterFirst(self.store,'tool_only'),seen),manifest,str(self.checkpoint),first,config,accepted_state=self.agent(harness=self.parent,manifest=manifest,config=config,next_cycle=0))
        agent=load_accepted_state(first/'cycle_00/state.json',manifest)
        proposer=DemoProposer(self.store,'tool_only');seen=[]
        result=run_outer_loop(self.components(proposer,seen),manifest,agent.checkpoint,self.root/'resumed',config,accepted_state=agent)
        self.assertEqual(proposer.requests,[(agent.checkpoint,agent.harness.version)])
        self.assertEqual(result['next_cycle'],2)
        self.assertEqual(result['protocol_hash'],agent.to_dict()['protocol_hash'])
        self.assertFalse((self.root/'resumed/cycle_00').exists())
        consumed=set(manifest.partitions['retirement_0']+manifest.partitions['test'])
        self.assertFalse(any(consumed.intersection(tasks) for _,_,tasks in seen))

    def test_historical_state_evolve_final_evaluation_executes_the_accepted_tool(self):
        manifest_path=self.root/'manifest.json';write_json(manifest_path,asdict(self.manifest))
        backend_path=self.root/'backend.json';write_json(backend_path,{'benchmark':'hotpotqa','evaluate':['fixture']})
        evolution=self.root/'evolution'
        result=run_outer_loop(self.components(DemoProposer(self.store,'tool_only'),[]),self.manifest,str(self.checkpoint),evolution,self.config,accepted_state=self.agent(harness=self.parent,next_cycle=0))
        module=script('evaluate_benchmark');observed=[]
        class Backend:
            def __init__(_,config,ledger):self.assertIn('--final-evaluation',config['evaluate'])
            def evaluate(_,checkpoint,harness,tasks,seeds,output):
                observed.append((checkpoint,harness.version,tasks))
                runner=InteractionTaskRunner(DemoEnvironment,lambda p:DemoModel(p,requires_diagnosis=False),max_steps=3)
                rollout=runner.rollout(checkpoint,harness,tasks,seeds=seeds,output=output)
                self.assertIn('log_query',rollout.trajectories[0].transitions[0].action)
                self.assertEqual(rollout.evaluations[0].success,1.)
                write_json(output/'response.json',{'benchmark_metrics':{'synthetic_success':1.},'cost':asdict(rollout.trajectories[0].cost)})
                return rollout.evaluations
        final=self.root/'final'
        argv=['evaluate_benchmark','--manifest',str(manifest_path),'--backend',str(backend_path),
            '--state',str(evolution/'deployment.json'),'--seeds','0','--output',str(final)]
        with patch.object(sys,'argv',argv),patch.object(module,'CommandBackend',Backend),contextlib.redirect_stdout(io.StringIO()):module.main()
        self.assertEqual(observed,[(result['checkpoint'],result['harness_revision']['version'],self.manifest.partitions['test'])])
        self.assertEqual(json.loads((final/'protocol.json').read_text())['agent']['harness_revision']['version'],result['harness_revision']['version'])

    def test_final_evaluator_requires_state_or_explicit_baseline(self):
        module=script('evaluate_benchmark')
        with patch.object(sys,'argv',['evaluate_benchmark','--manifest','x','--backend','x','--checkpoint','x','--output','x']),contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:module.main()
        self.assertEqual(error.exception.code,2)

    def test_explicit_baseline_is_the_only_way_to_load_an_empty_harness(self):
        from argparse import Namespace
        from internalization.core.accepted_state import evaluation_agent
        args=Namespace(state=None,baseline=True,checkpoint=self.checkpoint,protocol=None)
        with self.assertRaisesRegex(ValueError,"initial_agent.json"):
            evaluation_agent(args,self.manifest)
        agent=evaluation_agent(args,replace(self.manifest,benchmark="lawbench"))
        self.assertEqual(agent.harness.version,Harness().version)
        self.assertEqual(agent.protocol['mode'],'explicit_baseline')
        args.state=self.state({'checkpoint':str(self.checkpoint),'harness_revision':{'bad':'data'}})
        with self.assertRaisesRegex(ValueError,'Malformed accepted Harness'):evaluation_agent(args,self.manifest)

    def test_cli_resumes_state_without_a_separate_checkpoint_or_budget_override(self):
        from internalization import cli
        path=self.state(self.agent(next_cycle=0).to_dict())
        manifest=self.root/'manifest.json';write_json(manifest,asdict(self.manifest))
        backend=self.root/'backend.json';write_json(backend,{})
        argv=['hi','run','--state',str(path),'--manifest',str(manifest),'--backend',str(backend),'--output',str(self.root/'cli-output')]
        with patch.object(sys,'argv',argv),patch('internalization.command_backend.CommandBackend',return_value=None),\
             patch('internalization.outer_loop.run_outer_loop',return_value={}) as run,contextlib.redirect_stdout(io.StringIO()):cli.main()
        self.assertEqual(run.call_args.args[2],str(self.checkpoint))
        self.assertEqual(run.call_args.args[4],self.config)
        self.assertEqual(run.call_args.kwargs['accepted_state'].harness.version,self.full.version)

    def test_import_cli_generates_complete_two_cycle_manifest(self):
        module=script('build_benchmark_manifest')
        def hotpot(count,split):
            return [{'_id':f'{split}-{i}','question':f'{split} question {i}','context':[['Title',['Fact']]],
                     'answer':'SECRET','supporting_facts':[['Title',0]]} for i in range(count)]
        source=self.root/'test.json';train=self.root/'train.json'
        write_json(source,hotpot(1,'test'));write_json(train,hotpot(210,'train'))
        output=self.root/'import'
        argv=['build_benchmark_manifest','--benchmark','hotpotqa','--source',str(source),'--train-source',str(train),
              '--revision','fixture-v1','--cycles','2','--output',str(output)]
        with patch.object(sys,'argv',argv),contextlib.redirect_stdout(io.StringIO()):module.main()
        TaskManifest.load(output/'manifest.json').validate_loop(2,versioned=True,cohort_minimum=30)
        self.assertEqual(json.loads((output/'manifest.json').read_text())['loop_cycles'],2)

    def test_native_lawbench_final_rejects_unsupported_code_before_loading_model(self):
        module=script('evaluate_lawbench')
        manifest=replace(self.manifest,benchmark='lawbench');mf=self.root/'law-manifest.json';write_json(mf,asdict(manifest))
        state=self.state(self.agent(manifest=manifest).to_dict())
        config=self.root/'law-config.json';write_json(config,{'benchmark':'lawbench','catalog':'unused','max_steps':1})
        argv=['evaluate_lawbench','--manifest',str(mf),'--config',str(config),'--state',str(state),'--output',str(self.root/'law-final')]
        with patch.object(sys,'argv',argv),patch.object(module,'FrozenHFBackend') as model:
            with self.assertRaisesRegex(ValueError,'does not yet execute code revisions'):module.main()
        model.assert_not_called()

    def test_appworld_final_uses_same_state_loader(self):
        module=script('evaluate_appworld')
        manifest=replace(self.manifest,benchmark='appworld',partitions={'test_normal':('abcdef0_1',)})
        mf=self.root/'app-manifest.json';write_json(mf,asdict(manifest))
        state=self.state(self.agent(manifest=manifest).to_dict());config=self.root/'app-config.json';write_json(config,{})
        captured=[]
        class Backend:
            def __init__(*args):pass
            def evaluate(_,checkpoint,harness,tasks,seeds,output):
                captured.append(harness.version)
                return [EpisodeResult(t,s,1.,Cost()) for t in tasks for s in seeds]
        argv=['evaluate_appworld','--manifest',str(mf),'--backend',str(config),'--state',str(state),
            '--partition','test_normal','--output',str(self.root/'app-final')]
        with patch.object(sys,'argv',argv),patch.object(module,'CommandBackend',Backend),contextlib.redirect_stdout(io.StringIO()):module.main()
        self.assertEqual(captured,[self.full.version])


if __name__=='__main__':unittest.main()
