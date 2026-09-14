"""Mock API decisions, host-bound patches, and actual sandboxed reduced execution."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from internalization.core.interfaces import ProposalRequest
from internalization.core.types import Cost, write_json
from internalization.evolution.code_proposer import CodeProposer
from internalization.harness.revision import FileEdit, InternalizationTarget, RevisionStore, text_hash
from internalization.revision_demo import ROOT, DemoEnvironment, DemoModel, improvement, reduction
from internalization.training.rollout import InteractionTaskRunner


class CodeProposerBindingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = RevisionStore(self.root / 'revisions')
        self.parent = self.store.import_directory(ROOT / 'examples/versioned_harness/base')

    def request(self, harness=None, name='proposal', count=1):
        return ProposalRequest('old-policy', harness or self.parent, ('search',), (), (), (),
                               0, count, self.root / name)

    def proposer(self, response):
        calls = []
        def transport(payload):
            calls.append(payload)
            return {'choices': [{'message': {'content': json.dumps(response)}}],
                    'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}
        return CodeProposer(self.store, model='mock', transport=transport), calls

    def target_proposer(self, full, **extra):
        return self.proposer({'target': {'removed_behavior': 'Bypass diagnosis; keep the log tool',
                                        'supervision_adapter': full.config['supervision'], **extra}})

    def test_path_content_only_edits_bind_updates_additions_and_deletions(self):
        files = self.parent.files()
        files['prompts/obsolete.txt'] = 'old text'
        parent = self.store.snapshot(files)
        changes = [{'path': 'prompts/system.txt', 'content': '新的完整提示\n'},
                   {'path': 'tools/example.py', 'content': 'def run(): return 7\n'},
                   {'path': 'prompts/obsolete.txt', 'content': None}]
        proposer, calls = self.proposer({'candidates': [{'patch': changes, 'rationale': 'Update tools and prompt'}]})
        candidate, = proposer.propose(self.request(parent))
        candidate.validate(parent)
        self.assertEqual([p.before_hash for p in candidate.patch],
                         [text_hash(files['prompts/system.txt']), None, text_hash('old text')])
        self.assertEqual(parent.files(), files)
        self.assertNotIn('prompts/obsolete.txt', candidate.full_revision.files())
        self.assertEqual(candidate.full_revision.files()['prompts/system.txt'], '新的完整提示\n')
        archived = json.loads((self.root / 'proposal/candidate_0.json').read_text())
        self.assertEqual(archived['patch'][0]['before_hash'], text_hash(files['prompts/system.txt']))
        self.assertNotIn('before_hash', calls[0]['messages'][0]['content'])
        self.assertEqual(proposer.last_cost.model_calls, 1)

    def test_bound_patch_cannot_be_replayed_against_another_parent(self):
        change = [{'path': 'prompts/system.txt', 'content': 'replacement'}]
        bound = self.store.bind_patch(self.parent, change)
        other = self.store.apply(self.parent, (FileEdit('prompts/system.txt', bound[0].before_hash, 'different base'),))
        with self.assertRaisesRegex(ValueError, 'base hash mismatch'):
            self.store.apply(other, bound)
        self.assertEqual(other.files()['prompts/system.txt'], 'different base')

    def test_parent_tampering_while_model_proposes_is_rejected(self):
        path = Path(self.parent.path) / 'prompts/system.txt'
        class TamperingProposer(CodeProposer):
            def request_json(_, contract, public, output):
                path.chmod(0o644)
                path.write_text('changed after the request')
                return {'candidates': [{'patch': [{'path': 'tools/new.py', 'content': 'pass'}], 'rationale': 'Add tool'}]}
        candidate, = TamperingProposer(self.store).propose(self.request())
        self.assertIn('content changed', candidate['reason'])
        self.assertEqual(len(list(self.store.root.iterdir())), 1)

    def test_invalid_changes_do_not_create_snapshots_or_mutate_parent(self):
        before = self.parent.files()
        invalid = [
            [{'path': '../trainer.py', 'content': 'pass'}],
            [{'path': 'tools/new.py', 'content': 'pass', 'before_hash': 'guessed'}],
            [{'path': 'tools/new.py', 'content': None}],
            [{'path': 'tools/new.py', 'content': 42}],
            [{'path': 'tools/new.py', 'content': 'pass'}] * 2,
            [{'path': 'tools/new.py', 'content': 'def invalid('}],
        ]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises((ValueError, SyntaxError)):
                self.store.apply(self.parent, self.store.bind_patch(self.parent, changes))
            self.assertEqual(self.parent.files(), before)
            self.assertEqual(len(list(self.store.root.iterdir())), 1)

    def test_selection_only_builds_deterministic_reduction_and_preserves_new_tool(self):
        full = improvement(self.store, self.parent).full_revision
        proposer, calls = self.target_proposer(full)
        target = proposer.propose_target(self.request(full, 'target'))
        target.validate_structure()
        expected = reduction(self.store, full)
        self.assertEqual(target.reduced_revision.version, expected.reduced_revision.version)
        again = proposer.propose_target(self.request(full, 'target_again'))
        self.assertEqual(target.reduced_revision.version, again.reduced_revision.version)
        self.assertNotEqual(target.reduced_revision.path, again.reduced_revision.path)
        self.assertNotEqual(target.reduced_revision.version, self.parent.version)
        self.assertEqual({p for p in full.files() if full.files()[p] != target.reduced_revision.files()[p]},
                         {'config/harness.json'})
        self.assertIsNotNone(full.config['supervision'])
        self.assertIsNone(target.reduced_revision.config['supervision'])
        public = json.loads(calls[0]['messages'][1]['content'])
        self.assertEqual(set(public), {'full_revision', 'files', 'supervision_adapter', 'search_history'})
        saved = json.loads((self.root / 'target/internalization_target.json').read_text())
        self.assertEqual(InternalizationTarget.from_dict(saved), target)
        checkpoint = self.root / 'model'
        write_json(checkpoint / 'mock_model.json', {'trained': True, 'broken': False})
        runner = InteractionTaskRunner(DemoEnvironment, DemoModel, max_steps=3)
        result = runner.rollout(str(checkpoint), target.reduced_revision, ('search',),
                                seeds=(0,), output=self.root / 'reduced_rollout')
        self.assertEqual(result.evaluations[0].success, 1)
        self.assertIn('log_query', result.trajectories[0].transitions[0].action)
        self.assertIn('LOG_QUERY_RESULT', result.trajectories[0].transitions[1].student_prompt)
        self.assertTrue(all('DIAGNOSIS' not in t.student_prompt for t in result.trajectories[0].transitions))

    def test_no_hook_skips_api_and_resets_previous_call_cost(self):
        proposer, calls = self.proposer({'target': None})
        proposer.last_cost = Cost(100, 20, 1, 1)
        self.assertIsNone(proposer.propose_target(self.request(name='target')))
        self.assertEqual(calls, [])
        self.assertEqual(proposer.last_cost, Cost())
        selection = json.loads((self.root / 'target/target_selection.json').read_text())
        self.assertEqual(selection, {'target': None, 'reason': 'no_supervision_hook', 'model_called': False})
        self.assertEqual(json.loads((self.root / 'target/cost.json').read_text())['model_calls'], 0)

    def test_model_can_decline_target_without_creating_reduced_snapshot(self):
        full = improvement(self.store, self.parent).full_revision
        before = set(self.store.root.iterdir())
        proposer, calls = self.proposer({'target': None})
        self.assertIsNone(proposer.propose_target(self.request(full)))
        self.assertEqual(len(calls), 1)
        self.assertEqual(set(self.store.root.iterdir()), before)

    def test_target_rejects_patches_wrong_hook_or_missing_behavior(self):
        full = improvement(self.store, self.parent).full_revision
        before = set(self.store.root.iterdir())
        for i, extra in enumerate(({'patch': []}, {'supervision_adapter': 'agent/main.py:run'},
                                    {'removed_behavior': ''}, {'removed_behavior': None})):
            proposer, _ = self.target_proposer(full, **extra)
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                proposer.propose_target(self.request(full, f'bad_target_{i}'))
            self.assertEqual(set(self.store.root.iterdir()), before)

    def test_model_selection_cannot_certify_environment_interaction_as_compatible(self):
        full = improvement(self.store, self.parent, unsupported=True).full_revision
        proposer, _ = self.target_proposer(full)
        target = proposer.propose_target(self.request(full))
        checkpoint = self.root / 'model'
        write_json(checkpoint / 'mock_model.json', {'trained': False, 'broken': False})
        runner = InteractionTaskRunner(DemoEnvironment, lambda p: DemoModel(p, requires_diagnosis=False), max_steps=3)
        verdict = runner.check_internalization(str(checkpoint), target, ('search',), output=self.root / 'compatibility')
        self.assertFalse(verdict['supported'])
        self.assertIn('new environment observation/effect', verdict['reason'])
        self.assertIn('tools/log_query.py', full.files())

    def test_standalone_no_hook_target_requires_no_api_configuration(self):
        request = self.root / 'request.json'
        write_json(request, {'stage': 'target', 'checkpoint': 'old-policy', 'harness': self.parent.to_dict(),
                            'task_ids': ['search'], 'trajectories': [], 'scores': [], 'history': [],
                            'cycle': 0, 'candidate_count': 1})
        response = self.root / 'standalone/response.json'
        env = {k: v for k, v in os.environ.items() if not k.startswith('HI_PROPOSER_')}
        env['PYTHONPATH'] = str(ROOT / 'src')
        result = subprocess.run([sys.executable, '-m', 'internalization.training.entrypoint', 'target',
                                 '--request', str(request), '--response', str(response)],
                                env=env, cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        value = json.loads(response.read_text())
        self.assertIsNone(value['target'])
        self.assertEqual(value['cost']['model_calls'], 0)


if __name__ == '__main__':
    unittest.main()
