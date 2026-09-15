import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_attribution as fixtures
from internalization.core.manifest import TaskManifest
from internalization.core.types import Cost, EpisodeResult
from internalization.evaluation.retirement import (
    RetirementPolicy, evaluate_model_acceptance, evaluate_retirement,
)
from internalization.outer_loop import LoopConfig, run_outer_loop


class ModelAcceptanceTests(unittest.TestCase):
    policy = RetirementPolicy(bootstrap_samples=200)

    def rows(self, scores, *, seeds=(0,), tokens=100, aux=2):
        if isinstance(scores, (int, float)): scores = [scores] * 30
        return [EpisodeResult(f"task-{i}", seed, score, Cost(tokens, 0, 1+aux, aux, 1, 1+aux))
                for i, score in enumerate(scores) for seed in seeds]

    def cells(self, *, c=1., d=1.):
        return {"A": self.rows(1.), "B": self.rows(0., tokens=40, aux=0),
                "C": self.rows(c), "D": self.rows(d, tokens=40, aux=0)}

    def test_three_independent_decisions(self):
        for c, d, expected in ((1., 1., ("accept", "retire")),
                               (1., 0., ("accept", "retain")),
                               (0., 0., ("rollback", "retain"))):
            with self.subTest(c=c, d=d):
                verdict = evaluate_retirement(self.cells(c=c, d=d), self.policy)
                self.assertEqual((verdict["model_decision"], verdict["module_decision"]), expected)
                self.assertEqual(verdict["decision"], verdict["module_decision"])
                self.assertEqual(verdict["retirement_accepted"], expected == ("accept", "retire"))

    def test_strict_margin_and_uncertainty_are_not_mean_only_rejection(self):
        policy = RetirementPolicy(performance_margin=0.125, bootstrap_samples=200)
        a = self.rows(1.)
        self.assertEqual(evaluate_model_acceptance(a, self.rows(0.875), policy)["decision"], "accept")
        self.assertEqual(evaluate_model_acceptance(a, self.rows(0.874), policy)["decision"], "rollback")
        # Negative mean exceeding epsilon, but an interval crossing zero.
        result = evaluate_model_acceptance(self.rows([0.]*14+[1.]*16),
                                           self.rows([1.]*14+[0.]*16), self.policy)
        self.assertLess(result["interval"]["mean"], -self.policy.performance_margin)
        self.assertGreater(result["interval"]["high"], 0)
        self.assertEqual(result["decision"], "accept")
        self.assertEqual(result["reason"], "no_significant_full_degradation")

    def test_insufficient_tasks_fail_closed_and_bad_pairs_raise(self):
        a, c = self.rows([1.], seeds=tuple(range(40))), self.rows([1.], seeds=tuple(range(40)))
        result = evaluate_model_acceptance(a, c, self.policy)
        self.assertEqual(result["decision"], "rollback")
        self.assertEqual(result["reason"], "insufficient_independent_tasks")
        self.assertEqual(result["interval"]["tasks"], 1)
        for bad in (c[:-1], c + [c[0]]):
            with self.assertRaises(ValueError): evaluate_model_acceptance(a, bad, self.policy)

    def test_cost_failure_retains_module_without_rejecting_model(self):
        cells = self.cells()
        cells["D"] = self.rows(1., tokens=1000, aux=0)
        verdict = evaluate_retirement(cells, self.policy)
        self.assertTrue(verdict["capability_preserved"])
        self.assertFalse(verdict["cost_improved"])
        self.assertEqual((verdict["model_decision"], verdict["module_decision"]), ("accept", "retain"))

    def test_rollback_overrides_an_otherwise_eligible_retirement(self):
        # D looks excellent even though training destroyed full-Harness C.
        verdict = evaluate_retirement(self.cells(c=0., d=1.), self.policy)
        self.assertTrue(verdict["retirement_eligible"])
        self.assertFalse(verdict["retirement_accepted"])
        self.assertEqual((verdict["model_decision"], verdict["module_decision"]), ("rollback", "retain"))
        self.assertEqual(verdict["intervals"]["C-A"]["high"], -1.)

    def test_outer_three_branches_persist_checkpoint_and_candidate_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            for i,(c,d,model,module) in enumerate(((1.,1.,'accept','retire'),(1.,0.,'accept','retain'),(0.,0.,'rollback','retain'))):
                output=Path(directory)/f'case-{i}'
                result,calls=fixtures.AttributionTests().run_case(output,[{'A':1.,'B':0.,'C':c,'D':d}])
                expected='after-1' if model=='accept' else 'before'
                self.assertEqual(len(calls['training']),1)
                self.assertEqual(result['checkpoint'],expected)
                self.assertEqual(fixtures.active_controls(result),[] if module=='retire' else ['m0_0'])
                for name in ('retirement.json','state.json'):
                    row=json.loads((output/'cycle_00'/name).read_text())
                    self.assertEqual((row['model_decision'],row['module_decision']),(model,module))
                state=json.loads((output/'cycle_00/state.json').read_text())
                self.assertEqual(state['checkpoint'],expected)
                self.assertEqual(state['before_checkpoint'],'before')
                self.assertEqual(state['proposed_checkpoint'],'after-1')
                candidate=json.loads((output/'cycle_00/candidate_0/candidate.json').read_text())
                self.assertTrue(candidate['patch']);self.assertTrue(candidate['parent_revision'])
                self.assertEqual(candidate['candidate_id'],state['candidate_id'])
                self.assertEqual(json.loads((output/'cycle_00/training/checkpoint.json').read_text())['snapshot'],'after-1')
                if module=='retain': self.assertEqual(state['harness_revision'],candidate['full_revision'])

    def test_next_cycle_uses_accepted_checkpoint_and_residual_harness_after_each_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=fixtures.AttributionTests().run_case(Path(directory)/'run',[
                {'A':1.,'B':0.,'C':1.,'D':0.},
                {'A':1.,'B':0.,'C':0.,'D':0.},
                {'A':1.,'B':0.,'C':1.,'D':1.},
                {'A':.5,'B':.5}])
            self.assertEqual([r.checkpoint for r in calls['proposals']],['before','after-1','after-1','after-3'])
            self.assertEqual([fixtures.active_controls(r.harness) for r in calls['proposals']],
                [[],['m0_0'],['m0_0','m1_0'],['m0_0','m1_0']])
            self.assertEqual([r[0] for r in calls['training']],['before','after-1','after-1'])
            self.assertEqual([r[1] for r in calls['training']],['before','after-1','after-1'])
            self.assertEqual(result['checkpoint'],'after-3')
            self.assertEqual(fixtures.active_controls(result),['m0_0','m1_0','m3_0'])
            for request in calls['proposals']:
                for hidden in ('rollback','model_decision','C-A','model_acceptance'):
                    self.assertNotIn(hidden,json.dumps(request.history))
            for cycle,label,model,modules,tasks,seeds in calls['evaluations']:
                if cycle==2 and label in ('baseline_search','baseline_dev','A','B'): self.assertEqual(model,'after-1')

    def test_invalid_custom_evaluator_rolls_back_without_losing_accepted_harness(self):
        class Evaluator:
            def __init__(self,result):self.result=result
            def evaluate(self,*a,**kw):return self.result
        with tempfile.TemporaryDirectory() as directory:
            for i,verdict in enumerate(({'decision':'retain'},{'model_decision':'rollback','module_decision':'retire'})):
                output=Path(directory)/f'case-{i}'
                with patch('internalization.revision_loop.PairedRetirementEvaluator',return_value=Evaluator(verdict)):
                    result,calls=fixtures.AttributionTests().run_case(output,[{'A':1.,'B':0.}])
                self.assertEqual(result['checkpoint'],'before')
                self.assertEqual(fixtures.active_controls(result),['m0_0'])
                self.assertEqual(result['archive'][0]['reason'],'training_or_audit_failed')
                self.assertEqual(result['archive'][0]['model_decision'],'rollback')
                self.assertIn('independent model/module',result['archive'][0]['detail'])
