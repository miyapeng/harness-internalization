import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from internalization.harness_modules import ControlModule, Harness
from internalization.manifest import TaskManifest
from internalization.opid_adapter import real_task_ids, verify_response_mask
from internalization.records import Cost, EpisodeResult, State
from internalization.retirement_eval import RetirementPolicy, evaluate_retirement
from internalization.teacher_harness import Completion, TeacherHarness, distillation_selected


class FakeModel:
    snapshot_id = "frozen-v1"
    def __init__(self): self.prompts = []
    def generate(self, prompt, *, purpose):
        self.prompts.append(prompt)
        return Completion("PRIVATE-GUIDANCE", Cost(10, 2, 1, 1))


class CoreTests(unittest.TestCase):
    def module(self, source=None):
        if source is None:
            source = ('NAME="m"\nKIND="planner"\nINSTRUCTION="reason"\nPERSISTENCE=1\n'
                      'def trigger(history, step):\n    return step == 0\n')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "module.py"
            path.write_text(source)
            return ControlModule.load(path)

    def test_no_teacher_guidance_in_public_state_and_persistence(self):
        model = FakeModel()
        teacher = TeacherHarness(model, Harness((self.module(),)))
        state = State("real-id", "e", 0, "PUBLIC")
        guide = teacher.advise(state)
        self.assertEqual(state.public_history, "PUBLIC")
        self.assertIn("PRIVATE-GUIDANCE", guide.teacher_prompt)
        self.assertTrue(distillation_selected(guide, "m"))
        guide1 = teacher.advise(State("real-id", "e", 1, "PUBLIC NEW"))
        self.assertEqual(guide1.triggered_modules, ())
        self.assertTrue(distillation_selected(guide1, "m"))
        self.assertNotIn("PRIVATE-GUIDANCE", model.prompts[-1])
        guide2 = teacher.advise(State("real-id", "e", 2, "PUBLIC NEXT"))
        self.assertFalse(distillation_selected(guide2, "m"))
        self.assertTrue(distillation_selected(guide2, "m", "all"))

    def test_teacher_refresh_rejected(self):
        model = FakeModel()
        teacher = TeacherHarness(model, Harness((self.module(),)))
        model.snapshot_id = "changed"
        with self.assertRaises(RuntimeError): teacher.advise(State("t", "e", 0, "public"))

    def test_arbitrary_code_is_never_executed(self):
        with self.assertRaises(ValueError):
            self.module('import os\nos.system("touch /tmp/should-not-exist")')
        source = self.module().source.replace("step == 0", 'True or __import__("os")')
        with self.assertRaises(ValueError): self.module(source)

    def test_response_tokens_only(self):
        self.assertEqual(verify_response_mask([1, 1, 0, 0]), 2)
        with self.assertRaises(ValueError): verify_response_mask([1, 0, 1])
        with self.assertRaises(ValueError): verify_response_mask([1, 2, 0])

    def test_no_placeholder_task_ids_or_split_overlap(self):
        with self.assertRaises(ValueError): real_task_ids([{"uid": "placeholder"}])
        self.assertEqual(real_task_ids([{"extra.gamefile": "train/task/game.tw-pddl"}]),
                         ["train/task/game.tw-pddl"])
        with self.assertRaises(ValueError):
            TaskManifest("alfworld", "v1", {"train": ("same",), "dev": ("same",)}).validate()
        with self.assertRaises(ValueError): TaskManifest("x", "v1", {}).partition("test")


class RetirementTests(unittest.TestCase):
    policy = RetirementPolicy(bootstrap_samples=200, min_tasks=30)
    def rows(self, score, tokens, aux):
        return [EpisodeResult(f"task-{i}", 0, score, Cost(tokens, 0, 1+aux, aux, 1, 1+aux))
                for i in range(40)]
    def cells(self):
        return {"A": self.rows(1, 100, 2), "B": self.rows(0, 40, 0),
                "C": self.rows(1, 100, 2), "D": self.rows(1, 40, 0)}
    def test_positive_evidence_retires(self):
        self.assertEqual(evaluate_retirement(self.cells(), self.policy)["decision"], "retire")
    def test_joint_degradation_does_not_retire(self):
        cells = self.cells()
        cells["C"], cells["D"] = self.rows(0, 100, 2), self.rows(0, 40, 0)
        self.assertEqual(evaluate_retirement(cells, self.policy)["decision"], "retain")
    def test_preexisting_redundancy_is_not_internalization(self):
        cells = self.cells()
        cells["B"] = self.rows(1, 40, 0)
        self.assertFalse(evaluate_retirement(cells, self.policy)["checks"]["module_was_useful"])
    def test_token_increase_prevents_retirement(self):
        cells = self.cells()
        cells["D"] = self.rows(1, 1000, 0)
        self.assertEqual(evaluate_retirement(cells, self.policy)["decision"], "retain")
    def test_unpaired_missing_and_duplicate_results_rejected(self):
        cells = self.cells()
        cells["D"].pop()
        with self.assertRaises(ValueError): evaluate_retirement(cells, self.policy)
        cells = self.cells()
        cells["D"].append(cells["D"][0])
        with self.assertRaises(ValueError): evaluate_retirement(cells, self.policy)
    def test_many_seeds_are_not_independent_tasks(self):
        cells = self.cells()
        for name, rows in cells.items():
            cells[name] = [EpisodeResult("one-task", i, r.success, r.cost) for i, r in enumerate(rows)]
        self.assertEqual(evaluate_retirement(cells, self.policy)["decision"], "retain")


if __name__ == "__main__": unittest.main()
