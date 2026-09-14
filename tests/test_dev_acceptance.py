"""Dev evidence reuse and data isolation; synthetic scores, unchanged bootstrap gates."""
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from internalization.core.interfaces import Components
from internalization.core.types import Cost, EpisodeResult, Journal, write_json, digest
from internalization.core.manifest import TaskManifest
from internalization.core.accepted_state import AcceptedAgentState
from internalization.core.trajectory import RolloutResult
from internalization.evolution.revision_search import dev_acceptance_policy, public_history
from internalization.evaluation.retirement import RetirementPolicy
from internalization.evaluation.attribution import AttributionPolicy
from internalization.outer_loop import LoopConfig, run_outer_loop
from internalization.harness.revision import RevisionStore
from internalization.revision_demo import DemoProposer, ROOT


class DevAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.store=RevisionStore(self.root/'revisions')
        self.parent=self.store.import_directory(ROOT/'examples/versioned_harness/base')
        self.manifest=TaskManifest('synthetic','fixture',{
            p:tuple(f'{p}-{i:02d}' for i in range(30)) for p in ('train','search','dev','retirement_0','test')})
        self.seen=[];self.rows=[]

    def components(self,mode='tool_only'):
        owner=self
        class Runner:
            def rollout(_,model,harness,tasks,*,seeds,output,training=False):
                owner.seen.append((model,harness.version,tasks,output))
                if any(t.startswith(('acceptance_','test')) for t in tasks):raise AssertionError('Forbidden evaluation')
                gain='tools/log_query.py' in harness.files()
                rows=[]
                for task in tasks:
                    score=float(gain)
                    if tasks[0].startswith('dev') and mode=='dev_fails':score=.5
                    if tasks[0].startswith('dev') and mode=='uncertain_dev':
                        # Positive mean but CI touches zero: the existing dev gate must reject.
                        score=float(gain and task.endswith('00'))
                    rows.extend(EpisodeResult(task,seed,score,Cost(10,1,1,tool_calls=1,latency_s=.1)) for seed in seeds)
                owner.rows.extend(rows)
                return RolloutResult((),tuple(rows))
            def check_internalization(*a,**kw):
                if mode=='unsupported':return {'supported':False,'reason':'unsupported fixture'}
                return {'supported':True}
        class Trainer:
            def train(*a,**kw):raise AssertionError('No training expected')
        proposer=DemoProposer(self.store,'mixed' if mode in ('unsupported','attribution_fails') else 'tool_only')
        return Components(proposer,Runner(),Trainer())

    def run_loop(self,mode='tool_only',manifest=None,accepted=None):
        components=self.components(mode)
        # These fixtures have a tool gain but no incremental module effect on retirement tasks.
        result=run_outer_loop(components,manifest or self.manifest,'old',self.root/'run',
            LoopConfig(cycles=1,total_train_steps=1,seeds=(0,)),
            initial_harness=None if accepted else self.parent,accepted_state=accepted)
        return result

    def record(self):return json.loads((self.root/'run/cycle_00/harness_acceptance.json').read_text())

    def test_dev_accepts_without_extra_evaluations_and_cost_is_not_double_counted(self):
        result=self.run_loop()
        self.assertEqual(result['archive'][0]['reason'],'accepted_without_internalization')
        self.assertNotEqual(result['harness_revision']['version'],self.parent.version)
        self.assertEqual(len(self.seen),6)  # parent search/dev + two candidates search/dev
        self.assertEqual(len(self.rows),180)
        total=sum((row.cost for row in self.rows),Cost())
        self.assertEqual((total.total_tokens,total.model_calls,total.tool_calls),(1980,180,180))
        stored=[row for file in (self.root/'run').rglob('episodes.json') for row in json.loads(file.read_text())]
        self.assertEqual(sum(row['cost']['model_calls'] for row in stored),180)
        record=self.record()
        self.assertTrue(record['passed']);self.assertTrue(record['dev_accepted'])
        self.assertEqual(record['decision_source'],'dev');self.assertEqual(record['additional_evaluation_calls'],0)
        self.assertEqual(record['task_ids'],list(self.manifest.partitions['dev']))
        self.assertEqual(record['parent_revision'],self.parent.version)
        self.assertEqual(record['full_revision'],result['harness_revision']['version'])
        self.assertEqual(record['checkpoint'],'old')
        self.assertEqual(record['policy'],dev_acceptance_policy(RetirementPolicy()))
        for key,name in (('parent_dev_results','parent'),('candidate_dev_results','candidate')):
            self.assertEqual(record[key],json.loads((self.root/'run/cycle_00'/record['evidence'][name]).read_text()))
        self.assertFalse((self.root/'run/cycle_00/acceptance_plus').exists())
        self.assertFalse((self.root/'run/cycle_00/acceptance_parent').exists())

    def test_best_scoring_candidate_still_requires_positive_paired_dev_gain(self):
        result=self.run_loop('dev_fails')
        self.assertEqual(result['harness_revision']['version'],self.parent.version)
        self.assertEqual(result['checkpoint'],'old')
        self.assertEqual(result['archive'][0]['reason'],'no_useful_candidate')
        self.assertFalse(self.record()['dev_accepted']);self.assertFalse(self.record()['passed'])
        self.assertEqual(self.record()['dev_gain']['mean'],0)

    def test_positive_mean_with_nonpositive_lower_bound_is_not_accepted(self):
        result=self.run_loop('uncertain_dev')
        record=self.record()
        self.assertGreater(record['dev_gain']['mean'],0)
        self.assertEqual(record['dev_gain']['low'],0)
        self.assertFalse(record['passed']);self.assertEqual(result['harness_revision']['version'],self.parent.version)

    def test_old_acceptance_partition_is_ignored_and_never_reallocated(self):
        manifest=replace(self.manifest,partitions={**self.manifest.partitions,'acceptance_0':('unused-legacy-task',)})
        path=self.root/'old-manifest.json';write_json(path,asdict(manifest));before=path.read_bytes()
        result=self.run_loop(manifest=TaskManifest.load(path))
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(result['manifest_hash'],manifest.fingerprint)
        self.assertFalse(any('unused-legacy-task' in tasks for _,_,tasks,_ in self.seen))
        self.assertEqual(len(self.seen),6)

    def test_unsupported_target_keeps_dev_accepted_full_without_ab_or_training(self):
        result=self.run_loop('unsupported')
        self.assertEqual(result['archive'][0]['reason'],'accepted_without_internalization')
        self.assertEqual(result['archive'][0]['detail'],'unsupported fixture')
        self.assertEqual(result['harness_revision']['version'],self.record()['full_revision'])
        self.assertEqual(result['checkpoint'],'old');self.assertEqual(len(self.seen),6)

    def test_ab_failure_keeps_accepted_full_and_does_not_train(self):
        result=self.run_loop('attribution_fails')
        self.assertEqual(result['archive'][0]['reason'],'attribution_failed')
        self.assertEqual(result['harness_revision']['version'],self.record()['full_revision'])
        self.assertEqual(result['checkpoint'],'old')
        self.assertEqual(len(self.seen),8)  # Only two additional A/B evaluations.
        self.assertFalse((self.root/'run/cycle_00/training').exists())
        visible=public_history(Journal(self.root/'run/events.jsonl'))
        serialized=json.dumps(visible)
        for forbidden in ('retirement_0-','test-','parent_dev_results','candidate_dev_results','dev_gain'):
            self.assertNotIn(forbidden,serialized)

    def test_old_protocol_resume_records_transition_without_editing_original(self):
        config=LoopConfig(cycles=1,total_train_steps=1,seeds=(0,))
        original={'mode':'versioned_code_selective_internalization','loop':asdict(config),
            'retirement':asdict(RetirementPolicy()),'attribution':asdict(AttributionPolicy()),
            'harness_acceptance':asdict(AttributionPolicy()),'manifest_hash':self.manifest.fingerprint}
        state=AcceptedAgentState('old',self.parent,self.manifest.fingerprint,original,0)
        path=self.root/'old-state.json';write_json(path,state.to_dict());before=path.read_bytes()
        result=self.run_loop(accepted=state)
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(state.protocol,original)
        self.assertEqual(result['protocol']['harness_acceptance'],dev_acceptance_policy(RetirementPolicy()))
        self.assertEqual(result['protocol']['acceptance_transition']['from_protocol_hash'],digest(original))
        self.assertEqual(result['protocol']['acceptance_transition']['effective_from_cycle'],0)


if __name__=='__main__':unittest.main()
