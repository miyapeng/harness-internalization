import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from internalization.core.interfaces import Components
from internalization.core.manifest import TaskManifest
from internalization.core.trajectory import RolloutResult
from internalization.core.types import Cost, EpisodeResult
from internalization.evaluation.attribution import AttributionPolicy, evaluate_attribution
from internalization.evaluation.retirement import RetirementPolicy
from internalization.evolution.archive import CandidateArchive
from internalization.evolution.candidate import Candidate
from internalization.harness.module import Harness
from internalization.outer_loop import LoopConfig, run_outer_loop


ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'NAME="m"\nKIND="planner"\nINSTRUCTION="guide"\nPERSISTENCE=0\ndef trigger(history, step):\n    return True\n'


class AttributionTests(unittest.TestCase):
    policy = AttributionPolicy(bootstrap_samples=100)

    def rows(self, scores, *, task="task", seeds=(0,)):
        return [EpisodeResult(f"{task}-{i}", seed, score, Cost())
                for i, score in enumerate(scores) for seed in seeds]

    def test_zero_negative_and_uncertain_positive_gains_are_rejected(self):
        for a, b in (([0.5]*30, [0.5]*30), ([0.4]*30, [0.5]*30),
                     ([1.]*16+[0.]*14, [0.]*16+[1.]*14)):
            with self.subTest(a=a):
                result = evaluate_attribution(self.rows(a), self.rows(b), self.policy)
                self.assertFalse(result["passed"])
                self.assertEqual(result["reason"], "no_external_contribution")
        self.assertGreater(result["delta_external"]["mean"], 0)
        self.assertLessEqual(result["delta_external"]["low"], 0)

    def test_fixed_gain_threshold_is_strict_and_positive_evidence_passes(self):
        policy = AttributionPolicy(min_external_gain=0.25, bootstrap_samples=100)
        b = self.rows([0.5]*30)
        self.assertFalse(evaluate_attribution(self.rows([0.75]*30), b, policy)["passed"])
        self.assertTrue(evaluate_attribution(self.rows([1.]*30), b, policy)["passed"])

    def test_many_seeds_do_not_replace_independent_tasks(self):
        a, b = self.rows([1.], seeds=tuple(range(40))), self.rows([0.], seeds=tuple(range(40)))
        result = evaluate_attribution(a, b, self.policy)
        self.assertFalse(result["passed"])
        self.assertEqual(result["delta_external"]["tasks"], 1)
        self.assertFalse(result["checks"]["enough_independent_tasks"])

    def test_unpaired_or_duplicate_results_fail_closed(self):
        a, b = self.rows([1.]*30), self.rows([0.]*30)
        for invalid in (b[:-1], b + [b[0]], self.rows([0.]*30, seeds=(1,))):
            with self.assertRaises(ValueError): evaluate_attribution(a, invalid, self.policy)

    def test_policy_file_loads_and_invalid_thresholds_are_rejected(self):
        self.assertEqual(AttributionPolicy.load(ROOT / "configs/attribution.json"), AttributionPolicy())
        for kwargs in ({"min_external_gain": -0.1}, {"min_external_gain": float("nan")},
                       {"confidence": 1.0}, {"min_tasks": 1}, {"bootstrap_samples": 99}):
            with self.assertRaises(ValueError): AttributionPolicy(**kwargs)
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / "policy.json"
            file.write_text(json.dumps({**asdict(self.policy), "min_external_gain": 0.5}))
            loaded = AttributionPolicy.load(file)
            self.assertFalse(evaluate_attribution(self.rows([0.75]*30), self.rows([0.5]*30), loaded)["passed"])

    def run_case(self, output, cells, *, attribution_policy=None):
        calls = {"proposals": [], "evaluations": [], "training": []}
        class Proposer:
            def propose(self, request):
                calls["proposals"].append(request)
                # Policy must be persisted before evaluation/proposal starts.
                if not (output / "attribution_policy.json").is_file(): raise AssertionError("Unpinned policy")
                return tuple(Candidate(SOURCE.replace('NAME="m"', f'NAME="m{request.cycle}_{i}"'),
                    request.harness.version, request.cycle, i) for i in range(request.count))
        class Runner:
            def rollout(self, model, harness, tasks, *, seeds, output, training=False):
                label = output.name
                cycle = next(int(part.split("_")[1]) for part in output.parts if part.startswith("cycle_"))
                calls["evaluations"].append((cycle, label, model, tuple(m.name for m in harness.modules), tasks, seeds))
                if label in ("A", "B", "C", "D"):
                    score = cells[cycle][label]
                else:
                    # Both candidates pass search/dev independently of A/B.
                    score = 1.0 if output.parent.name.startswith("candidate_") else 0.0
                reduced = label in ("B", "D")
                cost = Cost(40 if reduced else 100, 0, 1 if reduced else 3,
                            0 if reduced else 2, 1, 1 if reduced else 3)
                return RolloutResult((), tuple(EpisodeResult(t, s, score, cost) for t in tasks for s in seeds))
        class Trainer:
            def train(self, student, teacher, full, reduced, trajectories, *, output, **kwargs):
                evidence = json.loads((output.parent / "attribution.json").read_text())
                if not evidence["passed"]: raise AssertionError("Training preceded contribution gate")
                calls["training"].append((student, teacher, full, reduced, kwargs))
                return f"after-{len(calls['training'])}"
        partitions = {p: tuple(f"{p}-{i}" for i in range(30))
                      for p in ("train", "search", "dev", "test", *(f"retirement_{i}" for i in range(len(cells))))}
        result = run_outer_loop(Components(Proposer(), Runner(), Trainer()), TaskManifest("mock", "v1", partitions),
            "before", output, LoopConfig(cycles=len(cells), total_train_steps=len(cells), seeds=(0,)),
            RetirementPolicy(bootstrap_samples=100), attribution_policy=attribution_policy or self.policy)
        return result, calls

    def test_equal_ab_never_trains_and_archives_discard_with_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result, calls = self.run_case(output, [{"A": 0.5, "B": 0.5}])
            self.assertEqual(calls["training"], [])
            self.assertFalse(any(row[1] in ("C", "D") for row in calls["evaluations"]))
            self.assertEqual(result["checkpoint"], "before")
            self.assertEqual(result["active_modules"], [])
            self.assertEqual(result["archive"][0]["decision"], "discard")
            self.assertEqual(result["archive"][0]["reason"], "no_external_contribution")
            self.assertFalse((output / "cycle_00/training").exists())
            self.assertFalse((output / "cycle_00/retirement.json").exists())
            state = json.loads((output / "cycle_00/state.json").read_text())
            self.assertEqual(state["checkpoint"], "before")
            archive = CandidateArchive(output / "candidate_archive.jsonl")
            attribution = next(r for r in archive.history() if r["kind"] == "attribution")
            self.assertEqual(attribution["parent"], Harness().version)
            self.assertEqual(attribution["candidate_id"], result["archive"][0]["candidate_id"])
            self.assertEqual(attribution["source"], result["archive"][0]["source"])
            events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1]["training_batches_spent"], 0)

    def test_positive_attribution_allows_training_and_four_cell_retirement(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result, calls = self.run_case(output, [{"A": 1., "B": 0., "C": 1., "D": 1.}])
            self.assertEqual(len(calls["training"]), 1)
            self.assertEqual(result["checkpoint"], "after-1")
            self.assertEqual(result["archive"][0]["decision"], "retire")
            before = [r for r in calls["evaluations"] if r[1] in ("A", "B")]
            self.assertEqual(len(before), 2)
            self.assertTrue(all(r[2] == "before" for r in before))
            self.assertEqual(before[0][4:], before[1][4:])
            self.assertEqual(len(before[0][3])-len(before[1][3]), 1)
            self.assertTrue((output / "cycle_00/retirement.json").is_file())

    def test_discard_keeps_residual_state_and_does_not_leak_attribution_to_proposer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            # First module useful but not internalized; subsequent modules have
            # no pre-training contribution. They cannot erase the retained one.
            result, calls = self.run_case(output, [
                {"A": 1., "B": 0., "C": 1., "D": 0.},
                {"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}])
            self.assertEqual(len(calls["training"]), 1)
            self.assertEqual(result["checkpoint"], "after-1")
            self.assertEqual(result["active_modules"], ["m0_0"])
            self.assertEqual([row["decision"] for row in result["archive"]], ["retain", "discard", "discard"])
            final_request = calls["proposals"][2]
            self.assertEqual(final_request.checkpoint, "after-1")
            self.assertEqual([m.name for m in final_request.harness.modules], ["m0_0"])
            self.assertEqual(len(final_request.history), 4)  # only two search candidates per prior cycle
            history = json.dumps(final_request.history)
            for forbidden in ("attribution", "no_external_contribution", "delta_external", "retirement_", "dev_gain"):
                self.assertNotIn(forbidden, history)

    def test_custom_fixed_policy_controls_actual_outer_training_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            policy = AttributionPolicy(min_external_gain=0.5, bootstrap_samples=100)
            result, calls = self.run_case(output, [{"A": 0.75, "B": 0.5}], attribution_policy=policy)
            self.assertEqual(calls["training"], [])
            self.assertEqual(result["checkpoint"], "before")
            self.assertEqual(json.loads((output / "attribution_policy.json").read_text()), asdict(policy))


if __name__ == "__main__": unittest.main()
