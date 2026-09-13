import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from internalization.harness_modules import ControlModule, Harness
from internalization.opid_adapter import OPIDProvider
from internalization.records import Cost, Journal, digest
from internalization.teacher_harness import Completion


@unittest.skipUnless(importlib.util.find_spec("torch"), "Optional CPU torch required for tensor bridge tests")
class OPIDBridgeTests(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        module_file = root / "m.py"
        module_file.write_text('NAME="m"\nKIND="planner"\nINSTRUCTION="PLAN"\nPERSISTENCE=0\n'
                               'def trigger(history, step):\n    return step == 0\n')
        class FakeBackend:
            snapshot_id = "frozen"
            def __init__(self): self.calls = []
            def assert_frozen(self): pass
            def prompt_ids(self, prompt): return [1, 2]
            def generate(self, prompt, *, purpose):
                self.calls.append(prompt)
                return Completion("teacher-private", Cost(3, 1, 1, 1))
            def score(self, prompt, response_ids):
                self.calls.append((prompt, response_ids))
                return [-0.5] * len(response_ids), Cost(5, 0, 1, 1)
        provider = object.__new__(OPIDProvider)
        provider.phase = {"supervision": "targeted"}
        provider.phase_hash = digest(provider.phase)
        provider.path = root / "phase.json"
        provider.path.write_text(json.dumps(provider.phase))
        provider.backend = FakeBackend()
        provider.harness = Harness((ControlModule.load(module_file),))
        provider.target = "m"
        provider.allowed_tasks = {"real-task"}
        provider.journal = Journal(root / "teacher.jsonl")
        self.provider = provider
        class Batch:
            def __init__(self):
                self.batch = {"responses": torch.tensor([[10, 11, 0], [12, 0, 0]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]]),
                    "prompts": torch.tensor([[1, 2], [1, 2]])}
                self.non_tensor_batch = {"obs_text": ["FIRST PUBLIC", "FUTURE OBSERVATION"],
                    "hi_task_id": ["real-task", "real-task"], "traj_uid": ["e", "e"], "step_num": [0, 1]}
            def __len__(self): return 2
        self.batch = Batch()

    def test_only_selected_student_tokens_receive_teacher_scores(self):
        original_prompts = self.batch.batch["prompts"].clone()
        original_responses = self.batch.batch["responses"].clone()
        self.provider.prepare(None, self.batch, {}, True)
        self.assertEqual(self.batch.batch["step_teacher_log_prob"].tolist(), [[-0.5, -0.5, 0], [0, 0, 0]])
        self.assertEqual(self.batch.batch["step_skill_mask"].tolist(), [True, False])
        self.assertTrue(self.torch.equal(self.batch.batch["prompts"], original_prompts))
        self.assertTrue(self.torch.equal(self.batch.batch["responses"], original_responses))
        self.assertNotIn("FUTURE", self.provider.backend.calls[0])
        self.assertNotIn("teacher-private", self.batch.non_tensor_batch["obs_text"][0])

    def test_input_parity_mismatch_fails_closed(self):
        self.batch.batch["prompts"][0, 0] = 999
        with self.assertRaises(ValueError): self.provider.prepare(None, self.batch, {}, True)

    def test_nontraining_task_fails_closed(self):
        self.batch.non_tensor_batch["hi_task_id"][0] = "held-out"
        with self.assertRaises(ValueError): self.provider.prepare(None, self.batch, {}, True)

    def test_teacher_error_propagates(self):
        def fail(prompt, ids): raise RuntimeError("scorer disconnected")
        self.provider.backend.score = fail
        with self.assertRaises(RuntimeError): self.provider.prepare(None, self.batch, {}, True)

    def test_phase_cannot_change_midbatch(self):
        self.provider.path.write_text('{"supervision":"all"}')
        with self.assertRaises(RuntimeError): self.provider.prepare(None, self.batch, {}, True)

    def test_teacher_recomputes_retained_modules_from_shared_raw_history(self):
        self.batch.non_tensor_batch["obs_text_base"] = ["FIRST PUBLIC", "FUTURE OBSERVATION"]
        self.batch.non_tensor_batch["obs_text"][0] = "FIRST PUBLIC\n\nstudent-retained-private"
        self.provider.prepare(None, self.batch, {}, True)
        self.assertNotIn("student-retained-private", self.provider.backend.calls[0])
        self.assertIn("FIRST PUBLIC", self.provider.backend.calls[0])

    def test_student_cannot_omit_shared_raw_history(self):
        self.batch.non_tensor_batch["obs_text_base"] = ["HIDDEN OLD HISTORY FIRST PUBLIC", "FUTURE OBSERVATION"]
        with self.assertRaises(ValueError): self.provider.prepare(None, self.batch, {}, True)


if __name__ == "__main__": unittest.main()
