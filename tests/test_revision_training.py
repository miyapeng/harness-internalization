"""CPU optimizer regression through real versioned-code rollout and supervision."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import test_behavior_policy as fixtures
from internalization.harness.revision import RevisionStore, FileEdit, text_hash
from internalization.harness.runtime import Completion
from internalization.revision_demo import ROOT, improvement, reduction, DemoEnvironment
from internalization.training.rollout import InteractionTaskRunner
from internalization.training.trainer import ModuleTrainer


@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU torch required')
class RevisionTrainingTests(unittest.TestCase):
    def run_training(self,root,*,noop=False,inactive=False,shared=False):
        policy=fixtures.BehaviorPolicyTests().policy()
        original=policy.generate
        policy.shared_calls=0
        def generate(prompt,*,purpose):
            if purpose=='harness_internal' and prompt=='shared computation':
                policy.shared_calls+=1
                result=original(prompt,purpose='planner')
                return Completion('SHARED_CONTEXT',result.cost)
            if purpose=='harness_internal': return original(prompt,purpose='planner')
            completion=original(prompt,purpose=purpose)
            action='finish' if 'LOG_QUERY_RESULT' in prompt else '{"tool":"log_query","arguments":{"needle":"action="}}'
            return Completion(action,completion.cost,completion.response_ids)
        policy.generate=generate
        store=RevisionStore(root/'revisions')
        parent=store.import_directory(ROOT/'examples/versioned_harness/base')
        full=improvement(store,parent).full_revision
        if shared:
            source=full.files()['agent/main.py'].replace('payload["history"]+"\\n"','payload["history"]+api.model("shared computation")+"\\n"')
            full=store.apply(full,(FileEdit('agent/main.py',text_hash(full.files()['agent/main.py']),source),))
        if noop or inactive:
            source='def augment(api,payload): return {"suffix":"","selected":'+str(not inactive)+'}\n'
            full=store.apply(full,(FileEdit('controls/diagnosis.py',text_hash(full.files()['controls/diagnosis.py']),source),))
        target=reduction(store,full)
        reference=fixtures.BehaviorPolicyTests().reference()
        trainer=ModuleTrainer(InteractionTaskRunner(DemoEnvironment,max_steps=3),tasks_per_batch=1,rollouts_per_task=2)
        trainer.train(policy,reference,full,target.reduced_revision,target=target,tasks=('train-0',),budget=2,output=root/'training')
        return policy

    def test_current_batch_alignment_response_masks_and_gradient_isolation(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            policy=self.run_training(Path(directory))
            self.assertEqual(policy.updates,2)
            pending=[];snapshots=[]
            for event in policy.events:
                if event[0]=='score': pending.append(event)
                if event[0]=='update':
                    self.assertEqual({e[1] for e in pending},{event[1]})
                    self.assertTrue(all(e[2]==(3,4) for e in pending))
                    snapshots.append(event[1]);pending=[]
            self.assertEqual(snapshots,['policy-0','policy-1'])
            for batch in policy.batches:
                self.assertEqual(tuple(batch['responses'].shape),(4,2))
                self.assertTrue(batch['response_mask'].all())
                self.assertEqual(batch['module_mask'].tolist(),[False,True,False,True])
                expected=torch.tensor([[0.,0.],[.0005,.0005],[0.,0.],[.0005,.0005]])
                torch.testing.assert_close(batch['advantages'],expected)
                self.assertFalse(batch['advantages'].requires_grad)

    def test_noop_difference_zero_after_real_cpu_parameter_updates(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            policy=self.run_training(Path(directory),noop=True)
            self.assertGreater(float(policy.model.weight.detach()[0,0]),-2)
            for batch in policy.batches:
                torch.testing.assert_close(batch['module_log_probs'],batch['old_log_probs'],rtol=0,atol=0)
                self.assertFalse(batch['advantages'].any())

    def test_inactive_hook_has_strictly_zero_teacher_term(self):
        with tempfile.TemporaryDirectory() as directory:
            policy=self.run_training(Path(directory),inactive=True)
            for batch in policy.batches:
                self.assertFalse(batch['module_mask'].any())
                self.assertFalse(batch['advantages'].any())

    def test_non_target_context_is_shared_without_regeneration_in_teacher(self):
        with tempfile.TemporaryDirectory() as directory:
            policy=self.run_training(Path(directory),shared=True)
            # Two batches * two trajectories * two steps. Teacher must not double this.
            self.assertEqual(policy.shared_calls,8)


if __name__=='__main__': unittest.main()
