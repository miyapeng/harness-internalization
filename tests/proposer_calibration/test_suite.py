import unittest
from .suite import CASE_TYPES, run_fake_calibration


class SyntheticCalibrationTests(unittest.TestCase):
    def test_twenty_synthetic_cases_share_host_outcomes(self):
        runs=[run_fake_calibration(name) for name in ('claude_code','codex','qwen_code')]
        self.assertEqual(len(CASE_TYPES),20)
        for run in runs:
            self.assertEqual(run['execution'],'fake_only')
            cases={row['case']:row for row in run['cases']}
            self.assertEqual(cases['null_target']['valid_candidates'],2)
            self.assertEqual(cases['invalid_target']['valid_candidates'],2)
            self.assertEqual(cases['invalid_target']['valid_declarations'],0)
            self.assertEqual(cases['named_control']['valid_declarations'],1)
            self.assertEqual(cases['read_only_boundary']['valid_candidates'],0)
            self.assertTrue(cases['read_only_boundary']['boundary_violation_detected'])
            self.assertEqual(cases['duplicate_candidates']['valid_candidates'],1)
            self.assertEqual(cases['task_id_resistance']['valid_candidates'],1)
            self.assertEqual(cases['unknown_evidence_task']['valid_candidates'],1)
        for name in ('valid_candidate_rate','proposal_completion_rate','distinct_candidate_rate',
                     'internalization_declaration_validity','boundary_violation_rate'):
            self.assertEqual(len({r['metrics'][name] for r in runs}),1)
        self.assertIsNone(runs[1]['metrics']['cost']);self.assertIsNone(runs[2]['metrics']['cost'])
