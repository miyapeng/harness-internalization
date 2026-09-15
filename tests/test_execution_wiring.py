from fixtures.code_training import pair
"""Task-one regressions: real file/process wiring, mock benchmark/model, CPU tensors."""
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

from internalization.command_backend import CommandBackend
from internalization.core.execution_config import resolve_execution, config_hash, request_execution, NoActorUpdates
from internalization.core.types import Cost, EpisodeResult
from internalization.core.interfaces import Components
from internalization.core.manifest import TaskManifest
from internalization.core.trajectory import RolloutResult, read_trace_file
from internalization.harness.revision import RevisionStore
from internalization.harness.code_runtime import CodeRuntime
from internalization.training.rollout import EnvironmentStep, InteractionTaskRunner
from internalization.outer_loop import LoopConfig, run_outer_loop
from internalization.revision_demo import DemoProposer, ROOT
import test_behavior_policy as fixtures


class ExecutionWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def backend_config(self,**execution):
        argv=['{python}',str(ROOT/'tests/fixtures/execution_worker.py'),'--request','{request}','--response','{response}']
        return {**{stage:argv for stage in ('propose','target','check_internalization','evaluate','train')},
                'execution':execution,'cwd':str(ROOT)}

    def test_internalization_requires_all_five_entrypoints_at_start(self):
        for missing in ('propose','target','check_internalization','evaluate','train'):
            config=self.backend_config();del config[missing]
            with self.subTest(missing=missing),self.assertRaisesRegex(ValueError,missing):
                CommandBackend(config,self.root/'ledger')
        config=self.backend_config(mode='evolution_only')
        for key in ('target','check_internalization','train'):del config[key]
        self.assertEqual(CommandBackend(config,self.root/'ledger').execution_config['mode'],'evolution_only')

    def test_cli_rejects_missing_entrypoint_before_manifest_or_model_loading(self):
        from internalization import cli
        config=self.backend_config();del config['target']
        path=self.root/'backend.json';path.write_text(json.dumps(config))
        with patch.object(sys,'argv',['hi','run','--manifest','MISSING','--backend',str(path),'--output',str(self.root/'out')]):
            with self.assertRaisesRegex(ValueError,'target'):cli.main()
        self.assertFalse((self.root/'out').exists())

    def test_unknown_fields_values_and_hash_mismatch_fail(self):
        for raw in ({'lambda':0},{'optimizer':{'lr':.1}},{'model':{'max_tokens':42}},
                    {'tasks_per_batch':False},{'supervision':'typo'},{'advantage':{'module_weight':float('nan')}}):
            with self.subTest(raw=raw),self.assertRaises(ValueError):resolve_execution(raw)
        with self.assertRaisesRegex(ValueError,'Unknown backend'):
            CommandBackend({**self.backend_config(),'typo':3},self.root/'ledger')
        config=resolve_execution({'advantage':{'module_weight':0}})
        self.assertEqual(config['advantage']['module_weight'],0)
        with self.assertRaisesRegex(ValueError,'mismatched'):
            request_execution({'effective_config':config,'effective_config_hash':'wrong'})

    def test_json_subprocess_reaches_real_trainer_scorer_model_loaders_and_batches(self):
        import torch
        # An inactive hook makes all vs targeted observable; weight=0 still leaves outcome RL.
        for mode,active,weight in (('targeted',False,.3),('all',False,.3),('targeted',True,0)):
            config=self.backend_config(device='cpu',reference_device='cpu',supervision=mode,
                tasks_per_batch=2,rollouts_per_task=3,max_steps=2,
                optimizer={'learning_rate':.023},advantage={'module_weight':weight})
            backend=CommandBackend(config,self.root/(mode+'.ledger'))
            full,target=pair(self.root/f'revisions-{active}',active=active)
            out=self.root/(mode+str(active))
            try: result=backend.train('initial',full,target.reduced_revision,target,('a','b','c'),2,out)
            except Exception:
                self.fail((out/'stderr.log').read_text())
            request=json.loads((out/'request.json').read_text())
            self.assertEqual(request['effective_config'],backend.execution_config)
            self.assertEqual(request['effective_config_hash'],config_hash(backend.execution_config))
            self.assertEqual(json.loads((out/'effective_config.json').read_text()),backend.execution_config)
            constructors=json.loads((out/'constructors.json').read_text())
            self.assertEqual(constructors['policy']['learning_rate'],.023)
            self.assertEqual(constructors['policy']['max_action_tokens'],512)
            self.assertEqual(constructors['reference']['max_context'],8192)
            summary=json.loads((out/'response.json').read_text())
            self.assertEqual([summary[k] for k in ('planned_update_batches','attempted_update_batches','actor_update_calls','optimizer_steps')],[2,2,2,2])
            self.assertEqual(summary['supervision'],mode)
            self.assertEqual(summary['advantage_config']['module_weight'],weight)
            samples=read_trace_file(out/'rollout_00000/trajectories.jsonl')
            self.assertEqual(len(samples),6);self.assertEqual({t.task_id for t in samples},{'a','b'})
            self.assertTrue(all(len(t.transitions)==2 for t in samples))
            rows=[json.loads(line) for line in (out/'training.jsonl').read_text().splitlines()]
            signals=[row for row in rows if row['kind']=='revision_teacher_state']
            self.assertTrue(signals)
            self.assertTrue(all(row['signal']['selected']==(active or mode=='all') for row in signals))
            self.assertTrue(Path(result).is_dir())
            evidence=json.loads((out/'batch_evidence.json').read_text())
            self.assertTrue(all(b['max_abs_advantage']==0 for b in evidence))
            if active:
                self.assertGreater(evidence[0]['module_log_probs'][0][0]-evidence[0]['old_log_probs'][0][0],.49)

    def test_no_update_reply_preserves_checkpoint_and_rejects_fake_new_model(self):
        backend=CommandBackend(self.backend_config(),self.root/'ledger')
        full,target=pair(self.root/'reply-revisions')
        reply={'planned_update_batches':2,'attempted_update_batches':2,'actor_update_calls':0,
               'optimizer_steps':None,'status':'no_actor_updates','checkpoint':'old','cost':asdict(Cost(tool_calls=2))}
        with patch.object(backend,'_call',return_value=reply):
            with self.assertRaises(NoActorUpdates):backend.train('old',full,target.reduced_revision,target,('t',),2,self.root/'train')
        with patch.object(backend,'_call',return_value={**reply,'checkpoint':'fake'}):
            with self.assertRaisesRegex(ValueError,'preserve'):backend.train('old',full,target.reduced_revision,target,('t',),2,self.root/'train')

    def test_verl_constructor_honors_optimizer_settings_and_counts_actual_steps(self):
        import torch
        import torch._dynamo  # Initialize before temporarily replacing external modules.
        from internalization.training.verl_backend import VerlPolicy
        from internalization.training.teacher_backend import FrozenHFBackend
        from types import SimpleNamespace,ModuleType
        modules={name:ModuleType(name) for name in ('omegaconf','verl','verl.workers','verl.workers.actor','verl.workers.actor.dp_actor')}
        modules['omegaconf'].OmegaConf=SimpleNamespace(create=lambda x:x)
        modules['verl.workers.actor.dp_actor'].DataParallelPPOActor=lambda *args:SimpleNamespace()
        def initialize(policy,*args,**kwargs):
            policy.torch=torch;policy.model=torch.nn.Linear(1,1);policy.model.gradient_checkpointing_enable=lambda:None
            policy.tokenizer=SimpleNamespace(pad_token_id=0);policy.snapshot_id='initial'
        with patch.dict(sys.modules,modules),patch('importlib.metadata.version',return_value='0.5.0'),\
             patch.object(FrozenHFBackend,'__init__',initialize),patch.object(torch.distributed,'is_initialized',return_value=True),\
             patch.object(torch.distributed,'get_world_size',return_value=1):
            policy=VerlPolicy('unused',device='cpu',learning_rate=.017,weight_decay=.03)
        self.assertEqual(policy.optimizer.param_groups[0]['lr'],.017)
        self.assertEqual(policy.optimizer.param_groups[0]['weight_decay'],.03)
        self.assertEqual(policy.optimizer_steps,0)
        for _ in range(3):
            policy.optimizer.zero_grad();policy.model(torch.ones(1,1)).sum().backward();policy.optimizer.step()
        self.assertEqual(policy.optimizer_steps,3)

    def test_zero_update_outer_loop_keeps_accepted_plus_and_old_policy(self):
        store=RevisionStore(self.root/'revisions');parent=store.import_directory(ROOT/'examples/versioned_harness/base')
        proposer=DemoProposer(store,'mixed')
        class Runner:
            def rollout(_,model,harness,tasks,*,seeds,output,training=False):
                # Tool gains admission; the control has independent pre-training contribution.
                score=float(bool(harness.config.get('supervision')))
                return RolloutResult((),tuple(EpisodeResult(t,s,score,Cost(10,1,1)) for t in tasks for s in seeds))
            def check_internalization(*a,**kw):return {'supported':True}
        class Trainer:
            def train(*a,**kw):raise NoActorUpdates({'planned_update_batches':1,'attempted_update_batches':1,'actor_update_calls':0,'optimizer_steps':None})
        partitions={name:tuple(f'{name}-{i}' for i in range(30)) for name in ('train','search','dev','test','acceptance_0','retirement_0')}
        result=run_outer_loop(Components(proposer,Runner(),Trainer(),execution_config=resolve_execution()),
            TaskManifest('mock','v1',partitions),'old',self.root/'loop',LoopConfig(cycles=1,total_train_steps=1,seeds=(0,)),initial_harness=parent)
        state=result['archive'][0]
        self.assertEqual(state['reason'],'no_actor_updates');self.assertEqual(result['checkpoint'],'old')
        self.assertNotEqual(result['harness_revision']['version'],parent.version)
        self.assertEqual(state['model_decision'],'unchanged');self.assertEqual(state['module_decision'],'retain')
        self.assertEqual(state['target']['full_revision']['version'],result['harness_revision']['version'])
        self.assertFalse((self.root/'loop/cycle_00/C').exists())
        self.assertEqual(json.loads((self.root/'loop/effective_config.json').read_text()),resolve_execution())

    def test_failed_rollout_records_attempts_not_requested_budget(self):
        from internalization.training.trainer import ModuleTrainer
        class Runner:
            def rollout(*args,**kwargs): raise RuntimeError('collection failed')
        trainer=ModuleTrainer(Runner())
        policy=fixtures.BehaviorPolicyTests().policy()
        full, target = pair(self.root/"revisions")
        with self.assertRaisesRegex(RuntimeError,'collection failed'):
            trainer.train(policy,fixtures.BehaviorPolicyTests().reference(),full,target.reduced_revision,
                tasks=('task',),target=target,budget=5,output=self.root/'failed')
        summary=json.loads((self.root/'failed/training_summary.json').read_text())
        self.assertEqual([summary[k] for k in ('planned_update_batches','attempted_update_batches','actor_update_calls')],[5,1,0])
        self.assertEqual(summary['status'],'failed')
        self.assertIsNone(summary['optimizer_steps'])
        self.assertFalse((self.root/'failed/checkpoint').exists())

    def test_explicit_evolution_only_does_not_call_target_checker_or_train(self):
        from test_revision_decisions import RevisionDecisionTests
        case=RevisionDecisionTests();case.setUp();self.addCleanup(case.doCleanups)
        original=run_outer_loop
        def run(components,*args,**kwargs):
            components.execution_config=resolve_execution({'mode':'evolution_only'})
            components.proposer.propose_target=lambda *a,**kw: self.fail('target requested in evolution_only')
            return original(components,*args,**kwargs)
        with patch('test_revision_decisions.run_outer_loop',run):
            result,_,_=case.run_loop('tool_only')
        self.assertEqual(result['archive'][0]['detail'],'evolution_only')
        self.assertEqual(result['checkpoint'],'old-model')

    def test_hotpot_versioned_check_evaluate_train_share_effective_limits(self):
        from dataclasses import replace
        from internalization.training import entrypoint,verl_backend
        from internalization.harness.revision import InternalizationTarget
        from internalization.command_backend import serialize_harness
        from test_benchmark_adapters import hotpot_config
        config=hotpot_config(self.root)
        env_path=self.root/'environment.json';env_path.write_text(json.dumps(asdict(config)))
        store=RevisionStore(self.root/'revisions');parent=store.import_directory(ROOT/'examples/versioned_harness/base')
        files=parent.files()
        files['controls/noop.py']='def run(api,p): return {"suffix":"","selected":False}\n'
        files['config/harness.json']=json.dumps({'schema':2,'entrypoint':parent.config['entrypoint'],
            'composition':'independent_suffix','controls':[{'id':'noop','entrypoint':'controls/noop.py:run','enabled':True}]})
        full=store.snapshot(files);target=InternalizationTarget.from_control(store,full,'noop','no-op control')
        effective=resolve_execution({'device':'cpu','reference_device':'cpu','max_steps':2,
            'tasks_per_batch':1,'rollouts_per_task':1,'supervision':'all','optimizer':{'learning_rate':.009}})
        payloads={
            'check_internalization':{'checkpoint':'old','target':target.to_dict(),'task_ids':['q1']},
            'evaluate':{'checkpoint':'old','harness':serialize_harness(full),'task_ids':['q1'],'seeds':[0]},
            'train':{'student_checkpoint':'old','teacher_checkpoint':'old','full_harness':serialize_harness(full),
                'reduced_harness':serialize_harness(target.reduced_revision),'target':target.to_dict(),
                'task_ids':['q1'],'planned_update_batches':2}}
        for stage,payload in payloads.items():
            out=self.root/stage;out.mkdir();request=out/'request.json';response=out/'response.json'
            request.write_text(json.dumps({'stage':stage,**payload,'effective_config':effective,'effective_config_hash':config_hash(effective)}))
            policy=fixtures.BehaviorPolicyTests().policy()
            # Eval is also eval/no-grad; the protected production HF loader normally guarantees this.
            policy.model.eval()
            original_generate=policy.generate
            def generate(prompt,*,purpose):
                import torch
                with torch.inference_mode():return original_generate(prompt,purpose=purpose)
            policy.generate=generate
            loader=(lambda *a,**kw:fixtures.BehaviorPolicyTests().reference()) if stage=='train' else (lambda *a,**kw:policy)
            with patch.object(sys,'argv',['worker','--request',str(request),'--response',str(response),
                    '--benchmark','hotpotqa','--env-config',str(env_path)]),patch.object(entrypoint,'FrozenHFBackend',side_effect=loader),                 patch.object(verl_backend,'VerlPolicy',return_value=policy) as policy_loader:
                entrypoint.main(stage)
            result=json.loads(response.read_text())
            if stage=='check_internalization':self.assertTrue(result['supported'],result)
            elif stage=='train':
                self.assertEqual(result['actor_update_calls'],2)
                self.assertEqual(policy_loader.call_args.kwargs['learning_rate'],.009)
            else:
                trajectories=read_trace_file(out/'trajectories.jsonl')
                self.assertEqual(len(trajectories[0].transitions),2)
                self.assertEqual(len(trajectories[0].environment_events),2)
            self.assertEqual(json.loads((out/'effective_config.json').read_text()),effective)
            protocol=json.loads((out/'benchmark_protocol.json').read_text())
            self.assertEqual(protocol['config']['max_steps'],2)



class ObservationWiringTests(unittest.TestCase):
    def run_events(self,kind,*,calls=1,prepare=False):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);store=RevisionStore(root/'revisions')
        parent=store.import_directory(ROOT/'examples/versioned_harness/base');files=parent.files()
        source='def run(api,p):\n    if p["operation"]=="prepare":\n'
        if prepare:source+='        api.environment("probe")\n'
        source+='        return {"prompt":p["history"],"tools":{},"memory":{}}\n'
        source+='    result=api.environment("act")\n'*calls
        source+='    return {"observation":result["observation"],"stop":False,"memory":{}}\n'
        files['agent/main.py']=source;revision=store.snapshot(files)
        class Env:
            def reset(self,*args):self.history='START';return self.history
            def step(self,action):
                self.history+='\nREPEATED ERROR'
                return EnvironmentStep(self.history if kind=='context' else 'REPEATED ERROR',1.,False,0.,observation_kind=kind)
            def close(self):pass
        from test_environment_events import EventPolicy
        result=InteractionTaskRunner(Env,max_steps=3).rollout(EventPolicy(),revision,('task',),seeds=(0,),output=root/'run')
        return result.trajectories[0]

    def test_full_context_replaces_history_and_environment_mirror_not_rendered(self):
        trajectory=self.run_events('context')
        prompts=[s.student_prompt for s in trajectory.transitions]
        self.assertEqual([p.count('START') for p in prompts],[1,1,1])
        self.assertEqual([p.count('REPEATED ERROR') for p in prompts],[0,1,2])
        self.assertFalse(any('[Harness tool result]' in p for p in prompts))
        self.assertEqual(trajectory.total_reward,3);self.assertEqual(len(trajectory.environment_events),3)
        self.assertTrue(all(e.observation_kind=='context' for e in trajectory.environment_events))

    def test_identical_delta_events_remain_independent_and_raw_audit_kept(self):
        trajectory=self.run_events('delta',calls=2)
        self.assertEqual([s.student_prompt.count('REPEATED ERROR') for s in trajectory.transitions],[0,2,4])
        self.assertEqual(len(trajectory.environment_events),6);self.assertEqual(trajectory.total_reward,6)
        self.assertEqual(len(trajectory.public_calls),6)
        self.assertEqual(trajectory.cost.tool_calls,6)
        self.assertTrue(all(c.result['observation']=='REPEATED ERROR' for c in trajectory.public_calls))

    def test_prepare_and_execute_occurrences_are_not_collapsed(self):
        trajectory=self.run_events('delta',prepare=True)
        self.assertEqual([s.student_prompt.count('REPEATED ERROR') for s in trajectory.transitions],[0,2,4])
        self.assertEqual([e.phase for e in trajectory.environment_events],['prepare','execute']*3)
        self.assertEqual(trajectory.total_reward,6)


if __name__=='__main__':unittest.main()
