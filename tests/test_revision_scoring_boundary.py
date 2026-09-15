"""Input/mask boundaries migrated from the retired phase/tensor bridge."""
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from fixtures.code_training import pair
from internalization.core.types import Cost, State
from internalization.core.trajectory import Transition, Trajectory
from internalization.harness.runtime import Completion
from internalization.training.teacher_scoring import ModuleTeacherScorer
from internalization.training.trainer import build_update_batch
from internalization.training.module_advantage import AdvantageConfig

@unittest.skipUnless(importlib.util.find_spec('torch'),'CPU torch required')
class RevisionScoringBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.full,self.target=pair(Path(self.tmp.name)/'revisions',condition='payload["step"] == 0')
        class Backend:
            snapshot_id='frozen'
            pad_token_id=0
            def __init__(self):self.calls=[]
            def assert_frozen(self):pass
            def prompt_ids(self,prompt):return [1,2]
            def generate(self,prompt,*,purpose):
                self.calls.append(prompt)
                return Completion('teacher-private',Cost(3,1,1,1))
            def score(self,prompt,ids):
                self.calls.append((prompt,ids))
                return [-.5]*len(ids),Cost(5,0,1,1)
        self.backend=Backend()
        steps=tuple(Transition(State('real-task','e',i,prompt),prompt,'act',ids,
                    old_log_probs=tuple(-1. for _ in ids),prompt_ids=(1,2))
                    for i,prompt,ids in ((0,'FIRST PUBLIC',(10,11)),(1,'FUTURE OBSERVATION',(12,))))
        self.trajectory=Trajectory('real-task','e',0,'frozen',self.target.reduced_revision.version,steps,0,Cost())
        self.scorer=ModuleTeacherScorer(self.backend,self.full,self.target,{'real-task'})
    def score(self):return self.scorer.score((self.trajectory,))
    def test_selected_tokens_padding_and_future_observation_isolation(self):
        original=self.trajectory
        signals,_=self.score()
        batch=build_update_batch((self.trajectory,),signals,self.backend,AdvantageConfig()).tensor_data
        self.assertEqual(batch['responses'].tolist(),[[10,11],[12,0]])
        self.assertEqual(batch['response_mask'].tolist(),[[1,1],[1,0]])
        self.assertEqual(batch['module_mask'].tolist(),[True,False])
        self.assertEqual(batch['module_log_probs'][0].tolist(),[-.5,-.5])
        self.assertEqual(batch['advantages'][1].tolist(),[0.,0.])
        self.assertFalse(batch['advantages'].requires_grad)
        self.assertEqual(self.trajectory,original)
        self.assertNotIn('FUTURE',self.backend.calls[0])
        self.assertNotIn('teacher-private',original.transitions[0].student_prompt)
    def test_prompt_tokenization_mismatch_fails_closed(self):
        step=replace(self.trajectory.transitions[0],prompt_ids=(999,2))
        self.trajectory=replace(self.trajectory,transitions=(step,))
        with self.assertRaisesRegex(ValueError,'tokenization'):self.score()
    def test_nontraining_task_fails_closed(self):
        self.scorer.allowed_tasks={'another-training-task'}
        with self.assertRaisesRegex(ValueError,'allowlist'):self.score()
    def test_teacher_error_propagates(self):
        def fail(prompt,ids):raise RuntimeError('scorer disconnected')
        self.backend.score=fail
        with self.assertRaisesRegex(RuntimeError,'disconnected'):self.score()
    def test_revision_cannot_change_midbatch(self):
        path=Path(self.full.path)/'controls/target.py'
        path.chmod(0o600)  # Simulate tampering only with this disposable test-owned file.
        path.write_text('def run(api,payload): return {}\n')
        with self.assertRaises(ValueError):self.score()
    def test_student_cannot_omit_shared_history(self):
        step=self.trajectory.transitions[0]
        step=replace(step,state=replace(step.state,public_history='HIDDEN OLD '+step.student_prompt))
        self.trajectory=replace(self.trajectory,transitions=(step,))
        with self.assertRaisesRegex(ValueError,'actual student context'):self.score()
    def test_non_target_context_is_reused_without_regeneration(self):
        step=self.trajectory.transitions[0]
        prompt=step.student_prompt+'\nstudent-retained-private'
        step=replace(step,student_prompt=prompt,state=replace(step.state,public_history=prompt))
        self.trajectory=replace(self.trajectory,transitions=(step,))
        self.score()
        self.assertIn('student-retained-private',self.backend.calls[0])
        self.assertEqual(len([c for c in self.backend.calls if isinstance(c,str)]),1)
