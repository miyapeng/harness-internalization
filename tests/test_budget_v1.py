"""Budget scheduling/data/worker contracts; synthetic CPU evidence, never a GPU run."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict

from internalization.core.execution_config import resolve_execution,request_execution,config_hash
from internalization.core.sampling import TaskQueue,search_schedule,BUDGET_SIZES,validate_budget_manifest,model_seed
from internalization.core.manifest import TaskManifest
from internalization.core.types import Cost,EpisodeResult,write_json,digest,Journal
from internalization.core.trajectory import Trajectory,RolloutResult
from internalization.core.interfaces import Components
from internalization.harness.revision import RevisionStore
from internalization.outer_loop import LoopConfig,run_outer_loop
from internalization.revision_demo import DemoProposer,ROOT
from internalization.benchmarks.budget_data import make_budget_manifest,grouped_selection


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)

    def test_three_strict_configs_and_seed_variants(self):
        for name,steps,context,prompt in [('alfworld',30,8192,4096),('webshop',15,8192,4096),('hotpotqa',8,16384,12288)]:
            config=json.loads((ROOT/f'configs/budget_v1/{name}.json').read_text())
            self.assertEqual(resolve_execution(config),config)
            self.assertEqual((config['max_steps'],config['model']['max_context'],config['model']['max_prompt_tokens']),(steps,context,prompt))
            self.assertEqual((config['tasks_per_batch'],config['rollouts_per_task']),(4,4))
            self.assertEqual(config['model']['max_new_tokens'],256)
            self.assertEqual(request_execution({'effective_config':config,'effective_config_hash':config_hash(config)}),config)
        for seed in (17,29,43):
            value=resolve_execution({'seeds':{'run_seed':seed,'model_sampling_seed':seed},'advantage':{'module_weight':0},'supervision':'all'})
            self.assertEqual(value['advantage']['module_weight'],0)
        with self.assertRaises(ValueError):resolve_execution({'seeds':{'typo_seed':42}})
        with self.assertRaises(ValueError):resolve_execution({'schedule':{'profile':'budget_v1','search_per_cycle':9}})

    def test_budget_cli_rejects_legacy_module_fallback(self):
        import sys
        from internalization import cli
        manifest=self.root/'manifest.json'
        write_json(manifest,{'benchmark':'alfworld','environment_revision':'fixture','split_seed':42,
            'partitions':{**{k:[f'{k}-{i}' for i in range(n)] for k,n in BUDGET_SIZES.items()},'test':['test']}})
        argv=['hi','run','--manifest',str(manifest),'--backend',str(ROOT/'configs/budget_v1/alfworld_backend.json'),
              '--checkpoint',str(self.root),'--output',str(self.root/'run')]
        with patch.object(sys,'argv',argv),self.assertRaisesRegex(ValueError,'executable Harness'):
            cli.main()
        self.assertFalse((self.root/'run').exists())

    def test_disjoint_search_and_resumable_distinct_training_batches(self):
        tasks=tuple(f't-{i}' for i in range(96))
        schedule=search_schedule(tasks,17)
        self.assertEqual([len(s) for s in schedule],[8]*3)
        self.assertEqual(len(set(sum(schedule,()))),24)
        self.assertEqual(schedule,search_schedule(tuple(reversed(tasks)),17))
        self.assertNotEqual(schedule,search_schedule(tasks,29))
        queue=TaskQueue(tasks[:9],17)
        draws=[]
        for i in range(10):
            batch=queue.take(4);self.assertEqual(len(set(batch)),4);draws.extend(batch)
            queue=TaskQueue(tasks[:9],17,json.loads(json.dumps(queue.state())))
        uninterrupted=TaskQueue(tasks[:9],17)
        self.assertEqual(draws,[t for _ in range(10) for t in uninterrupted.take(4)])
        self.assertEqual(queue.state()['draws'],40)
        self.assertNotEqual(model_seed(17,0,'t',0),model_seed(17,0,'t',1))
        with self.assertRaises(ValueError):TaskQueue(tasks[:10],17,queue.state())

    def run_loop(self,dev_gain=True,search_gain=True):
        store=RevisionStore(self.root/'store');parent=store.import_directory(ROOT/'examples/versioned_harness/base')
        manifest=TaskManifest('synthetic','fixture',{**{k:tuple(f'{k}-{i}' for i in range(n)) for k,n in BUDGET_SIZES.items()},'test':('test',)})
        validate_budget_manifest(manifest)
        seen=[];requests=[]
        base_proposer=DemoProposer(store,'tool_only')
        class Proposer:
            def propose(_,request):
                requests.append(request)
                return base_proposer.propose(request)
            def propose_target(_,request):return None
        class Runner:
            def rollout(_,model,harness,tasks,*,seeds,output,training=False):
                seen.append((tasks,output,seeds,harness.version))
                self.assertFalse(any(t.startswith(('test','retirement','acceptance')) for t in tasks))
                gain='tools/log_query.py' in harness.files()
                score=float(gain and (dev_gain if tasks[0].startswith('dev') else bool(search_gain)))
                episodes=tuple(Trajectory(t,f'{t}:0',0,'fixed-model',harness.version,(),
                    score if search_gain!='weak' or tasks[0].startswith('dev') or t==tasks[0] else 0.,Cost(1,1,1)) for t in tasks)
                journal=Journal(output/'trajectories.jsonl')
                for trajectory in episodes: journal.append('trajectory',trajectory=asdict(trajectory))
                return RolloutResult(episodes,tuple(t.outcome for t in episodes))
        class Trainer:
            def train(*args,**kwargs):raise AssertionError('No target must not train')
        cfg=resolve_execution({'schedule':{'profile':'budget_v1'},'rollouts_per_task':4})
        components=Components(Proposer(),Runner(),Trainer(),execution_config=cfg)
        result=run_outer_loop(components,manifest,'old',self.root/'run',LoopConfig(seeds=(0,)),initial_harness=parent)
        return seen,requests,result,parent

    def test_eight_scores_four_traces_single_dev_finalist_no_acceptance(self):
        seen,requests,result,parent=self.run_loop()
        self.assertTrue(all(len(r.tasks)==8 and len(r.scores)==8 and len(r.trajectories)==4 for r in requests))
        self.assertEqual(len(set(sum((r.tasks for r in requests),()))),24)
        first=[s for s in seen if 'cycle_00' in str(s[1])]
        self.assertEqual([len(s[0]) for s in first],[8,8,8,32,32])
        self.assertEqual(sum(len(s[0]) for s in first),88)
        self.assertTrue(all(s[2]==(0,) for s in seen))
        self.assertEqual(result['checkpoint'],'old')
        self.assertNotEqual(result['harness_revision']['version'],parent.version)
        record=json.loads((self.root/'run/cycle_00/harness_acceptance.json').read_text())
        self.assertTrue(record['passed']);self.assertEqual(record['decision_source'],'dev')
        self.assertFalse(record['search_gain']['significance_test'])
        from internalization.core.trajectory import read_trace_file
        self.assertEqual(len(read_trace_file(self.root/'run/cycle_00/baseline_search/trajectories.jsonl')),8)
        self.assertEqual(len(json.loads((self.root/'run/cycle_00/baseline_search/episodes.json').read_text())),8)

    def test_search_positive_mean_does_not_require_significance(self):
        self.run_loop(search_gain='weak')
        record=json.loads((self.root/'run/cycle_00/harness_acceptance.json').read_text())
        self.assertEqual(record['search_gain']['mean'],.125)
        self.assertTrue(record['accepted'] if 'accepted' in record else record['passed'])

    def test_webshop_official_source_ranges_are_kept(self):
        rows=[]
        for i in range(4000):
            source='official_test' if i<500 else 'official_eval' if i<1500 else 'official_train'
            rows.append({'id':f'webshop:{i}','source':source,'source_hash':'fixture',
                'split':'test' if i<500 else 'dev' if i<1500 else 'train',
                'task_type':str((i//2)%3),'group_id':str(i//2)})
        manifest,report=make_budget_manifest('webshop','fixture',rows)
        for name,ids in manifest.partitions.items():
            actual=[int(t.split(':')[1]) for t in ids]
            if name=='test':self.assertEqual(actual,list(range(500)))
            elif name in ('train','search'):self.assertTrue(min(actual)>=1500)
            else:self.assertTrue(min(actual)>=500 and max(actual)<1500)

    def test_no_positive_search_skips_dev_and_training(self):
        seen,_,result,parent=self.run_loop(search_gain=False)
        self.assertEqual(len(seen),9)
        self.assertTrue(all(len(s[0])==8 for s in seen))
        self.assertEqual(result['harness_revision']['version'],parent.version)

    def test_failed_dev_does_not_try_second_finalist(self):
        seen,_,result,parent=self.run_loop(dev_gain=False)
        self.assertEqual(sum('dev' in str(s[1].name) for s in seen),6)
        self.assertEqual(result['harness_revision']['version'],parent.version)

    def test_exact_grouped_data_counts_and_source_roles(self):
        rows=[]
        for source,count in [('official_train',2700),('valid_seen',8),('valid_unseen',8)]:
            rows += [{'id':f'{source}-{i}','source':source,'source_hash':'fixture','split':'train' if source=='official_train' else 'test',
                'group_id':f'{source}-group-{i//2}','task_type':str((i//2)%3)} for i in range(count)]
        manifest,report=make_budget_manifest('alfworld','fixture-source',rows)
        validate_budget_manifest(manifest)
        assigned={}
        for partition,ids in manifest.partitions.items():
            for row in rows:
                if row['id'] in ids:
                    self.assertIn(assigned.setdefault(row['group_id'],partition),(partition,))
        self.assertEqual(report['requested_sizes'],BUDGET_SIZES)
        self.assertFalse(any(k.startswith('acceptance') for k in manifest.partitions))
        with self.assertRaises(ValueError):grouped_selection(rows[:4],3,42)
        with self.assertRaises(ValueError):make_budget_manifest('alfworld','fixture-source',rows[:10])

    def test_webshop_reset_reward_and_hidden_goal_boundary(self):
        from internalization.benchmarks.webshop import WebShopEnvironment
        from internalization.benchmarks.common import BenchmarkConfig
        secret={'asin':'hidden-asin','product_category':'fixture','secret':'GOLD_SECRET'}
        catalog=self.root/'catalog.json'
        write_json(catalog,{'benchmark':'webshop','revision':'fixture','tasks':[{'id':'webshop:1500','split':'train',
            'session_id':1500,'source':'official_train','source_hash':'fixture','goal_hash':digest(secret)}]})
        resets=[]
        class Env:
            def __init__(self):self.server=type('Server',(),{'goals':{1500:secret}})()
            def reset(self,*,session):resets.append(session);return 'Public instruction',None
            def get_available_actions(self):return {'has_search_bar':True,'clickables':['buy now']}
            def step(self,action):return 'GOLD_SECRET',.75,True,None
            def close(self):pass
        env=WebShopEnvironment(BenchmarkConfig('webshop',str(catalog)),self.root/'env',training=True,environment_loader=lambda options:Env())
        self.assertNotIn('GOLD',env.reset('webshop:1500',3)['observation'])
        result=env.step('click[Buy Now]')
        self.assertEqual(resets,[1500]);self.assertEqual(result['reward'],.75)
        self.assertEqual(result['observation_kind'],'context');self.assertNotIn('GOLD',result['observation'])
        self.assertEqual(json.loads((self.root/'env/grade.json').read_text())['metrics']['reward'],.75)

    def test_subprocess_config_and_queue_continue_across_training_phases(self):
        from internalization.command_backend import CommandBackend
        from internalization.harness.module import Harness,HarnessModule
        from internalization.core.trajectory import read_trace_file
        import test_behavior_policy as fixtures
        argv=['{python}',str(ROOT/'tests/fixtures/execution_worker.py'),'--request','{request}','--response','{response}']
        backend=CommandBackend({**{s:argv for s in ('propose','target','check_internalization','evaluate','train')},
            'cwd':str(ROOT),'execution':{'schedule':{'profile':'budget_v1'},'device':'cpu','reference_device':'cpu',
                'rollouts_per_task':4,'max_steps':1,'seeds':{'environment_seed':73,'run_seed':29,'model_sampling_seed':43},
                'optimizer':{'learning_rate':.002},'advantage':{'module_weight':0}}},self.root/'costs')
        full=Harness((HarnessModule.from_source(fixtures.SOURCE),))
        tasks=tuple('task-'+str(i) for i in range(32))
        all_tasks=[]
        for phase in range(2):
            output=self.root/f'phase-{phase}'
            try: backend.train('initial',full,Harness(),'target',tasks,2,output)
            except Exception:
                self.fail((output/'stderr.log').read_text())
            rows=[json.loads(line) for line in (output/'training.jsonl').read_text().splitlines()]
            for batch in (r for r in rows if r['kind']=='batch_tasks'):
                self.assertEqual(len(set(batch['task_ids'])),4);all_tasks.extend(batch['task_ids'])
            samples=read_trace_file(output/'rollout_00000/trajectories.jsonl')
            self.assertEqual(len(samples),16);self.assertTrue(all(t.seed==73 for t in samples))
            seed_rows=[json.loads(line) for line in (output/'rollout_00000/trajectories.jsonl').read_text().splitlines()]
            self.assertEqual(len({r['model_sampling_seed'] for r in seed_rows if r['kind']=='episode_seed'}),16)
            summary=json.loads((output/'response.json').read_text())
            self.assertEqual(summary['advantage_config']['module_weight'],0)
            self.assertEqual(summary['sampling_state']['draws'],8*(phase+1))
            constructors=json.loads((output/'constructors.json').read_text())
            self.assertEqual(constructors['policy']['learning_rate'],.002)
            self.assertEqual(json.loads((output/'seed_config.json').read_text())['environment_seed'],73)
        self.assertEqual(len(set(all_tasks)),16)
        self.assertEqual(backend.sampling_state['draws'],16)

if __name__=='__main__':unittest.main()
