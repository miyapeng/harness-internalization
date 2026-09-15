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
from internalization.evolution.candidate import HarnessCandidate
from internalization.harness.revision import HarnessRevision, RevisionStore, InternalizationTarget
from fixtures.code_training import AGENT
from internalization.outer_loop import LoopConfig, run_outer_loop


ROOT = Path(__file__).resolve().parents[1]



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
        store=RevisionStore(output.parent/(output.name+"-revisions"))
        initial=store.snapshot({'agent/main.py':AGENT,'config/harness.json':json.dumps({
            'schema':2,'entrypoint':'agent/main.py:run','composition':'independent_suffix','controls':[]})})
        class Proposer:
            def propose(self, request):
                calls["proposals"].append(request)
                if not (output / "protocol.json").is_file(): raise AssertionError("Unpinned policy")
                result=[]
                for i in range(request.count):
                    cid=f'm{request.cycle}_{i}'
                    config=request.harness.config
                    config['controls'].append({'id':cid,'entrypoint':f'controls/{cid}.py:run','enabled':True})
                    changes=[{'path':'config/harness.json','content':json.dumps(config)},
                        {'path':f'controls/{cid}.py','content':'def run(api,payload): return {"suffix":"diagnostic","selected":True}\n'}]
                    result.append(HarnessCandidate.create(store,request.harness,store.bind_patch(request.harness,changes),cid))
                return tuple(result)
            def propose_target(self,request):
                cid=request.harness.config['controls'][-1]['id']
                return InternalizationTarget.from_control(store,request.harness,cid,cid)
        class Runner:
            def rollout(self, model, harness, tasks, *, seeds, output, training=False):
                label = output.name
                cycle = next(int(part.split("_")[1]) for part in output.parts if part.startswith("cycle_"))
                calls["evaluations"].append((cycle, label, model, tuple(active_controls(harness)), tasks, seeds))
                if label in ("A", "B", "C", "D"): score = cells[cycle][label]
                else: score = 1.0 if output.parent.name.startswith("candidate_") else 0.0
                reduced = label in ("B", "D")
                cost = Cost(40 if reduced else 100, 0, 1 if reduced else 3,
                            0 if reduced else 2, 1, 1 if reduced else 3)
                return RolloutResult((), tuple(EpisodeResult(t, s, score, cost) for t in tasks for s in seeds))
            def check_internalization(self,*args,**kwargs): return {'supported':True}
        class Trainer:
            def train(self, student, teacher, full, reduced, trajectories, *, output, **kwargs):
                evidence = json.loads((output.parent / "attribution.json").read_text())
                if not evidence["passed"]: raise AssertionError("Training preceded contribution gate")
                calls["training"].append((student, teacher, full, reduced, kwargs))
                output.mkdir();(output/'checkpoint.json').write_text(json.dumps({'snapshot':f"after-{len(calls['training'])}"}))
                return f"after-{len(calls['training'])}"
        partitions = {p: tuple(f"{p}-{i}" for i in range(30))
                      for p in ("train", "search", "dev", "test", *(f"retirement_{i}" for i in range(len(cells))))}
        result = run_outer_loop(Components(Proposer(), Runner(), Trainer()), TaskManifest("mock", "v1", partitions),
            "before", output, LoopConfig(cycles=len(cells), total_train_steps=len(cells), seeds=(0,)),
            RetirementPolicy(bootstrap_samples=100), attribution_policy=attribution_policy or self.policy,initial_harness=initial)
        return result, calls

    def test_equal_ab_never_trains_but_preserves_dev_accepted_revision(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result, calls = self.run_case(output, [{"A": 0.5, "B": 0.5}])
            self.assertEqual(calls["training"], [])
            self.assertFalse(any(row[1] in ("C", "D") for row in calls["evaluations"]))
            self.assertEqual(result["checkpoint"], "before")
            self.assertEqual(active_controls(result), ['m0_0'])
            self.assertEqual(result['archive'][0]['reason'],'attribution_failed')
            self.assertEqual(result['archive'][0]['model_decision'],'unchanged')
            self.assertFalse((output / "cycle_00/training").exists())
            self.assertFalse((output / "cycle_00/retirement.json").exists())
            candidate=json.loads((output/'cycle_00/candidate_0/candidate.json').read_text())
            self.assertEqual(candidate['candidate_id'],result['archive'][0]['candidate_id'])
            self.assertEqual(candidate['parent_revision'],calls['proposals'][0].harness.version)
            self.assertEqual(candidate['full_revision']['version'],result['harness_revision']['version'])
            self.assertTrue(json.loads((output/'cycle_00/harness_acceptance.json').read_text())['passed'])

    def test_positive_attribution_allows_training_and_four_cell_retirement(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result, calls = self.run_case(output, [{"A": 1., "B": 0., "C": 1., "D": 1.}])
            self.assertEqual(len(calls["training"]), 1)
            self.assertEqual(result["checkpoint"], "after-1")
            self.assertEqual(result["archive"][0]["module_decision"], "retire")
            before = [r for r in calls["evaluations"] if r[1] in ("A", "B")]
            self.assertEqual(len(before), 2)
            self.assertTrue(all(r[2] == "before" for r in before))
            self.assertEqual(before[0][4:], before[1][4:])
            self.assertEqual(len(before[0][3])-len(before[1][3]), 1)
            self.assertTrue((output / "cycle_00/retirement.json").is_file())

    def test_failed_attribution_keeps_residual_state_and_does_not_leak_to_proposer(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            result, calls = self.run_case(output, [
                {"A": 1., "B": 0., "C": 1., "D": 0.},
                {"A": 0.5, "B": 0.5}, {"A": 0.5, "B": 0.5}])
            self.assertEqual(len(calls["training"]), 1)
            self.assertEqual(result["checkpoint"], "after-1")
            self.assertEqual(active_controls(result), ["m0_0","m1_0","m2_0"])
            self.assertEqual([row["reason"] for row in result["archive"]], ['internalization_audited','attribution_failed','attribution_failed'])
            final_request = calls["proposals"][2]
            self.assertEqual(final_request.checkpoint, "after-1")
            self.assertEqual(active_controls(final_request.harness), ["m0_0","m1_0"])
            self.assertEqual(len(final_request.history), 4)
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
            self.assertEqual(json.loads((output / "protocol.json").read_text())['attribution'], asdict(policy))


def active_controls(value):
    revision=HarnessRevision.from_dict(value['harness_revision']) if isinstance(value,dict) else value
    return [c['id'] for c in revision.config['controls'] if c['enabled']]
