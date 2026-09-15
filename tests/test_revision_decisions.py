"""Fast orchestration-only mocks supplement the actual sandbox/lifecycle tests."""
import json
import tempfile
import unittest
from pathlib import Path

from internalization.core.interfaces import Components
from internalization.core.manifest import TaskManifest
from internalization.core.trajectory import RolloutResult
from internalization.core.types import Cost, EpisodeResult
from internalization.evolution.code_proposer import CodeProposer
from internalization.harness.revision import RevisionStore, FileEdit, InternalizationTarget, text_hash
from internalization.outer_loop import LoopConfig, run_outer_loop
from internalization.revision_demo import ROOT, DemoProposer, improvement, reduction


class RevisionDecisionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.store=RevisionStore(self.root/'revisions')
        self.parent=self.store.import_directory(ROOT/'examples/versioned_harness/base')

    def run_loop(self,mode,*,cycles=1):
        partitions={name:tuple(f'{name}-{i}' for i in range(30)) for name in
            ('train','search','dev','test',*(f'acceptance_{i}' for i in range(cycles)),*(f'retirement_{i}' for i in range(cycles)))}
        manifest=TaskManifest('mock','v1',partitions)
        proposer=DemoProposer(self.store,'tool_only');seen=[]
        class Runner:
            def rollout(_,model,harness,tasks,*,seeds,output,training=False):
                seen.append((model,harness.version,tasks))
                success=float('tools/log_query.py' in harness.files())
                if mode=='no_gain' or (mode=='reject_dev' and tasks[0].startswith('dev')):success=0.
                return RolloutResult((),tuple(EpisodeResult(t,s,success,Cost(10,1,1)) for t in tasks for s in seeds))
        class Trainer:
            def train(*a,**kw):raise AssertionError('Trainer must not run')
        result=run_outer_loop(Components(proposer,Runner(),Trainer()),manifest,'old-model',self.root/'loop',
            LoopConfig(cycles=cycles,total_train_steps=cycles,seeds=(0,)),initial_harness=self.parent)
        return result,proposer,seen

    def test_no_gain_keeps_old_model_and_parent(self):
        result,_,_=self.run_loop('no_gain')
        self.assertEqual(result['harness_revision']['version'],self.parent.version)
        self.assertEqual(result['archive'][0]['reason'],'no_useful_candidate')

    def test_search_gain_cannot_bypass_dev_acceptance_gate(self):
        result,_,_=self.run_loop('reject_dev')
        self.assertEqual(result['archive'][0]['reason'],'no_useful_candidate')
        self.assertEqual(result['harness_revision']['version'],self.parent.version)

    def test_valid_tool_without_target_persists_into_next_candidate_search(self):
        result,proposer,seen=self.run_loop('tool_only',cycles=2)
        accepted=result['archive'][0]['harness_revision']['version']
        self.assertEqual(proposer.requests,[('old-model',self.parent.version),('old-model',accepted)])
        self.assertEqual(result['checkpoint'],'old-model')
        self.assertEqual(result['harness_revision']['version'],accepted)
        self.assertFalse(any(t.startswith(('retirement','test')) for _,_,tasks in seen for t in tasks))
        self.assertEqual(result['archive'][1]['reason'],'no_useful_candidate')

    def test_null_embedded_target_keeps_accepted_candidate(self):
        result,_,_=self.run_loop('tool_only')
        state=result['archive'][0]
        self.assertEqual(state['reason'],'accepted_without_internalization')
        self.assertEqual(state['detail'],'no_embedded_internalization_target')
        self.assertNotEqual(result['harness_revision']['version'],self.parent.version)

    def test_supervision_rejects_removal_of_non_target_tool_registration(self):
        full=improvement(self.store,self.parent).full_revision
        target=reduction(self.store,full)
        minus=self.store.apply(target.reduced_revision,(FileEdit('config/tools.json',text_hash(target.reduced_revision.files()['config/tools.json']),'{}'),))
        with self.assertRaisesRegex(ValueError,'shared runtime/tool/context'):
            InternalizationTarget(full,minus,'diagnosis',full.config['supervision']).validate_structure()

    def test_proposer_transport_is_independently_mockable_and_bad_patch_does_not_lose_sibling(self):
        from internalization.core.interfaces import ProposalRequest
        candidate=improvement(self.store,self.parent,diagnosis=False)
        class MockProposer(CodeProposer):
            def request_json(self,contract,public,output):
                output.mkdir(parents=True)
                self.public=public
                return {'candidates':[{'patch':[{'path':'../trainer.py','content':'pass'}],'rationale':'bad','evidence_refs':[],'internalization':None},
                    {'patch':[{"path":p.path,"content":p.content} for p in candidate.patch],'rationale':'new tool','evidence_refs':[],'internalization':None}]}
        proposer=MockProposer(self.store)
        request=ProposalRequest('old',self.parent,('search',),(),(),(),0,2,self.root/'proposals')
        proposals=proposer.propose(request)
        self.assertIn('invalid_proposal',proposals[0])
        proposals[1].validate(self.parent)
        self.assertEqual(proposer.public['files'],self.parent.files())
        self.assertNotIn('retirement',proposer.public)


if __name__=='__main__': unittest.main()
