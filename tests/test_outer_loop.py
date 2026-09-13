import tempfile
import unittest
from pathlib import Path

from internalization.demo import ToyBackend
from internalization.manifest import TaskManifest
from internalization.outer_loop import LoopConfig, run_outer_loop
from internalization.records import Cost, EpisodeResult
from internalization.retirement_eval import RetirementPolicy


class OuterLoopTests(unittest.TestCase):
    def test_retained_module_survives_and_later_cycles_continue(self):
        class RetainFirstBackend(ToyBackend):
            def evaluate(self, checkpoint, harness, tasks, seeds, output):
                rows = super().evaluate(checkpoint, harness, tasks, seeds, output)
                if checkpoint == "toy:checkpoint:1" and not harness.modules:
                    # Deliberately make the first reduced deployment expensive.
                    return [EpisodeResult(r.task_id, r.seed, r.success, Cost(10000, 0, 1, 0, 1, 1)) for r in rows]
                return rows
        partitions = {p: tuple(f"{p}-{i}" for i in range(30))
            for p in ("train", "search", "dev", "retirement_0", "retirement_1", "retirement_2")}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result = run_outer_loop(RetainFirstBackend(), TaskManifest("toy", "v1", partitions),
                "toy:initial", output, LoopConfig(total_train_steps=90, seeds=(0,)),
                RetirementPolicy(bootstrap_samples=100, min_token_saving_fraction=0))
            self.assertEqual(len(result["archive"]), 3)
            self.assertEqual(result["archive"][0]["decision"], "retain")
            self.assertIn("planner_0", result["active_modules"])
            self.assertTrue((output / "cycle_02/state.json").exists())


if __name__ == "__main__": unittest.main()
