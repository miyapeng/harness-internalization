import tempfile
import unittest
from pathlib import Path

from internalization.harness_modules import ControlModule, Harness
from internalization.records import Cost, Journal
from internalization.student_harness import StudentHarness
from internalization.teacher_harness import Completion


class StudentHarnessTests(unittest.TestCase):
    def test_retained_modules_run_without_target_and_keep_public_history(self):
        class Actor:
            snapshot_id = "current-actor"
            def __init__(self): self.purposes = []
            def generate(self, prompt, *, purpose):
                self.purposes.append(purpose)
                return Completion("retained-advice", Cost(10, 1, 1, 1))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            modules = []
            for name, kind in (("keep", "planner"), ("remove", "review")):
                path = root / f"{name}.py"
                path.write_text(f'NAME="{name}"\nKIND="{kind}"\nINSTRUCTION="guide"\nPERSISTENCE=0\n'
                                'def trigger(history, step):\n    return True\n')
                modules.append(ControlModule.load(path))
            actor = Actor()
            student = StudentHarness(actor, Harness(tuple(modules)).without("remove"), Journal(root / "trace.jsonl"))
            original = {"text": ["PUBLIC HISTORY", "DONE HISTORY"], "image": None}
            augmented = student.augment(original, ["a", "b"], ["e0", "e1"], 0, [True, False])
            self.assertEqual(actor.purposes, ["planner"])
            self.assertEqual(original["text"], ["PUBLIC HISTORY", "DONE HISTORY"])
            self.assertEqual(augmented["text_base"], original["text"])
            self.assertTrue(augmented["text"][0].startswith("PUBLIC HISTORY"))
            self.assertIn("retained-advice", augmented["text"][0])
            self.assertEqual(augmented["text"][1], "DONE HISTORY")


if __name__ == "__main__": unittest.main()
