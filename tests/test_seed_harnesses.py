"""Scripted public fixtures; real revision sandbox, no model/API/GPU training."""
from argparse import Namespace
from dataclasses import asdict, replace
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from internalization.benchmarks.alfworld import AlfworldEnvironment
from internalization.benchmarks.common import BenchmarkConfig
from internalization.benchmarks.isolated import environment_factory
from internalization.benchmarks.webshop import WebShopEnvironment
from internalization.benchmarks.webshop_projection import webshop_action
from internalization.core.accepted_state import evaluation_agent, load_accepted_state
from internalization.core.interfaces import Components
from internalization.core.seed_harnesses import bind_initial_seed, SEEDED_BENCHMARKS
from internalization.core.trajectory import RolloutResult
from internalization.core.types import Cost, EpisodeResult, digest, write_json
from internalization.harness.revision import RevisionStore
from internalization.harness.runtime import Completion
from internalization.outer_loop import LoopConfig, run_outer_loop
from internalization.training.rollout import EnvironmentStep, InteractionTaskRunner
from test_agent_state_pipeline import records
from internalization.benchmarks.importers import make_catalog_manifest
from test_benchmark_adapters import hotpot_config

ROOT = Path(__file__).resolve().parents[1]


class ScriptedModel:
    snapshot_id = 'scripted-h0-policy'

    def __init__(self, responses):
        self.responses = responses
        self.prompts, self.scored = [], []

    def generate(self, prompt, *, purpose):
        if purpose not in ('action', 'rollout_action'):
            raise AssertionError('H0 must not invoke an auxiliary model')
        index = len(self.prompts)
        self.prompts.append(prompt)
        return Completion(self.responses[index], Cost(5, 3, 1), (101, 200 + index, 303))

    def score(self, prompt, response_ids):
        self.scored.append((prompt, tuple(response_ids)))
        return [-.5] * len(response_ids), Cost()

    def prompt_ids(self, prompt):
        return [1, 2]


class DictEnvironment:
    """Only translate the worker wire contract; use actual benchmark logic."""
    def __init__(self, environment): self.environment = environment
    def reset(self, task, seed): return self.environment.reset(task, seed)['observation']
    def step(self, action): return EnvironmentStep(**self.environment.step(action))
    def close(self): self.environment.close()


class SeedHarnessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = RevisionStore(self.root / 'revisions')
        self.seeds = {name: self.store.import_directory(ROOT / 'seed_harnesses' / name)
                      for name in SEEDED_BENCHMARKS}

    def run_fixture(self, name, factory, task, responses, *, harness=None, suffix='rollout'):
        model = ScriptedModel(responses)
        revision = harness or self.seeds[name]
        runner = InteractionTaskRunner(factory, max_steps=len(responses))
        result = runner.rollout(model, revision, (task,), seeds=(0,),
                                output=self.root / suffix, training=True)
        trajectory = result.trajectories[0]
        self.assertEqual(trajectory.harness_version, revision.version)
        self.assertEqual(trajectory.model_version, model.snapshot_id)
        self.assertEqual(len(trajectory.transitions), len(responses))
        self.assertEqual(trajectory.cost.model_calls, len(responses))
        self.assertEqual(trajectory.cost.tool_calls, sum(e.tool_calls for e in trajectory.environment_events))
        for index, transition in enumerate(trajectory.transitions):
            self.assertEqual(transition.action, responses[index])
            self.assertEqual(transition.response_ids, (101, 200 + index, 303))
            self.assertEqual(model.scored[index], (transition.student_prompt, transition.response_ids))
            self.assertEqual(transition.old_log_probs, (-.5, -.5, -.5))
            self.assertNotIn('[Harness tool result]', transition.student_prompt)
            self.assertNotIn('SECRET', transition.student_prompt)
        return result, model

    def test_registered_seed_identity_and_foreign_generic_smoke_rejection(self):
        generic = self.store.import_directory(ROOT / 'examples/versioned_harness/base')
        smoke = self.store.import_directory(ROOT / 'examples/budget_v1_smoke')
        for name, seed in self.seeds.items():
            self.assertEqual(seed.config['schema'], 2)
            self.assertEqual(seed.config['controls'], [])
            record = bind_initial_seed(name, seed)
            self.assertEqual(record['revision_hash'], seed.version)
            self.assertEqual(record['benchmark'], name)
            for other in [generic, smoke] + [s for n, s in self.seeds.items() if n != name]:
                with self.assertRaisesRegex(ValueError, 'matching seed_harnesses'):
                    bind_initial_seed(name, other)
            changed = self.store.snapshot({**seed.files(), 'prompts/new.txt': 'changed'})
            with self.assertRaisesRegex(ValueError, 'modified H0'):
                bind_initial_seed(name, changed)

    def test_provenance_is_protected_and_adapter_drift_fails_before_run(self):
        seed = self.seeds['alfworld']
        self.assertNotIn('configs/seed_harnesses.json', seed.files())
        with self.assertRaisesRegex(ValueError, 'outside configured'):
            self.store.bind_patch(seed, [{'path': '../configs/seed_harnesses.json', 'content': '{}'}])
        registry = json.loads((ROOT / 'configs/seed_harnesses.json').read_text())
        registry['seeds']['alfworld']['fixed_adapter_sha256']['src/internalization/benchmarks/alfworld.py'] = 'wrong'
        path = self.root / 'bad-registry.json'; write_json(path, registry)
        with patch('internalization.core.seed_harnesses.REGISTRY', path):
            with self.assertRaisesRegex(ValueError, 'adapter changed'):
                bind_initial_seed('alfworld', seed)

    def test_cli_requires_matching_seed_before_backend_execution(self):
        from internalization import cli
        checkpoint = self.root / 'model'; checkpoint.mkdir()
        for name in SEEDED_BENCHMARKS:
            _, manifest = make_catalog_manifest(name, 'fixture', records(1, 'test'), records(150), cycles=1)
            manifest_path = self.root / (name + '.json'); write_json(manifest_path, asdict(manifest))
            backend_path = self.root / (name + '-backend.json'); write_json(backend_path, {'benchmark': name})
            argv = ['hi', 'run', '--manifest', str(manifest_path), '--backend', str(backend_path),
                '--checkpoint', str(checkpoint), '--cycles', '1', '--planned-update-batches', '1',
                '--output', str(self.root / (name + '-run')), '--revision-store', str(self.root / 'cli-revisions'),
                '--harness-workspace', str(ROOT / 'seed_harnesses' / name)]
            with patch.object(sys, 'argv', argv), patch('internalization.command_backend.CommandBackend', return_value=None), \
                 patch('internalization.outer_loop.run_outer_loop', return_value={}) as run, \
                 contextlib.redirect_stdout(io.StringIO()):
                cli.main()
            self.assertEqual(run.call_args.kwargs['initial_harness'].version, self.seeds[name].version)
            argv[-1] = str(ROOT / 'examples/versioned_harness/base')
            with patch.object(sys, 'argv', argv), patch('internalization.command_backend.CommandBackend', return_value=None), \
                 patch('internalization.outer_loop.run_outer_loop') as run:
                with self.assertRaisesRegex(ValueError, 'matching seed_harnesses'):
                    cli.main()
                run.assert_not_called()

    def test_alfworld_seed_preserves_golden_public_prompt_history_and_raw_tokens(self):
        fixture = json.loads((ROOT / 'tests/fixtures/alfworld_prompt_reference.json').read_text())
        task = self.root / 'game.tw-pddl'; task.touch()
        native_actions = []
        class Native:
            def seed(self, value): pass
            def reset(self):
                return [fixture['observations'][0]], {'extra.gamefile': [str(task)],
                    'admissible_commands': [fixture['admissible']], 'hidden': 'SECRET'}
            def step(self, actions):
                native_actions.append(actions[0]); index = len(native_actions)
                done = index == len(fixture['actions'])
                return [fixture['observations'][index]], [999], [done], {
                    'won': [done], 'admissible_commands': [fixture['admissible']], 'hidden': 'SECRET'}
            def close(self): pass
        native = Native()
        class ExternalPackage:
            def __init__(self, config, train_eval): pass
            def init_env(self, batch_size): return native
        responses = ['<think>Use the public action.</think><action>' + action.upper() + '</action>'
                     for action in fixture['actions']]
        result, model = self.run_fixture('alfworld', lambda: AlfworldEnvironment(
            ROOT / 'configs/alfworld.yaml', environment_class=ExternalPackage), str(task), responses)
        self.assertEqual(native_actions, fixture['actions'])
        self.assertEqual(model.prompts, fixture['expected'][:-1])
        self.assertEqual(result.trajectories[0].total_reward, 10.)
        self.assertNotIn('JSON', model.prompts[0])
        self.assertNotIn("'help'", model.prompts[0])

    def test_webshop_seed_public_actions_tag_parser_history_and_official_reward(self):
        secret = {'asin': 'SECRET_ASIN', 'answer': 'SECRET_GOAL'}
        catalog = self.root / 'webshop.json'
        write_json(catalog, {'benchmark': 'webshop', 'revision': 'fixture', 'tasks': [{
            'id': 'webshop:1500', 'split': 'train', 'session_id': 1500, 'source': 'official_train',
            'source_hash': 'fixture', 'goal_hash': digest(secret)}]})
        native_actions, sessions = [], []
        class Native:
            server = SimpleNamespace(goals={1500: secret})
            def reset(self, *, session):
                sessions.append(session)
                return 'Public task: buy a blue mug. Public page 0.', {'hidden': 'SECRET'}
            def get_available_actions(self):
                if len(native_actions) == 3: raise AssertionError('Terminal debug controls read')
                return {'has_search_bar': True, 'clickables': ['public-product', 'buy now'], 'hidden': 'SECRET'}
            def step(self, action):
                native_actions.append(action)
                if len(native_actions) == 3: return 'SECRET terminal debug', .75, True, {'hidden': 'SECRET'}
                return 'Public page ' + str(len(native_actions)) + '.', 0., False, {}
            def close(self): pass
        environment = WebShopEnvironment(BenchmarkConfig('webshop', str(catalog)), self.root / 'env',
            training=True, environment_loader=lambda _: Native())
        responses = ['<think>Search.</think>\n<action>search[Blue Mug]</action>',
            '<think>Inspect.</think><action>click[PUBLIC-PRODUCT]</action>',
            '<think>Purchase.</think><action>click[Buy Now]</action>']
        result, model = self.run_fixture('webshop', lambda: DictEnvironment(environment), 'webshop:1500', responses)
        self.assertEqual(sessions, [1500])
        self.assertEqual(native_actions, ['search[blue mug]', 'click[public-product]', 'click[buy now]'])
        self.assertEqual(result.trajectories[0].total_reward, .75)
        for index, prompt in enumerate(model.prompts):
            self.assertEqual(prompt.count('You are an expert autonomous agent'), 1)
            self.assertEqual(prompt.count('Public task:'), 1)
            for page in range(index + 1): self.assertEqual(prompt.count(f'Public page {page}.'), 1)
            self.assertIn("'search[<your query>]'", prompt)
            self.assertIn("'click[public-product]'", prompt)
        self.assertTrue(all(t.action_valid for t in result.trajectories[0].transitions))
        self.assertEqual(webshop_action('search[Blue Mug]'), 'search[Blue Mug]')
        self.assertEqual(webshop_action('<action>click[Buy Now]</action>'), 'click[buy now]')
        bad = '<action>search[x]</action><action>click[y]</action>'
        self.assertEqual(webshop_action(bad), bad)

    def test_hotpot_seed_executes_three_json_actions_in_worker(self):
        config = hotpot_config(self.root)
        # Four equally searchable documents distinguish the unchanged top-k=3
        # retrieval contract from accidentally exposing the whole distractor set.
        catalog = json.loads(Path(config.catalog).read_text())
        catalog['tasks'][0]['record']['context'] += [['Tower B', ['PUBLIC_B']], ['Tower C', ['PUBLIC_C']],
                                                    ['Tower Z', ['OMITTED_BY_TOP_K']]]
        config = replace(config, catalog=str(self.root / 'hotpot-top-k.json'))
        write_json(Path(config.catalog), catalog)
        responses = ['{"action":"search","query":"tower"}', '{"action":"lookup","title":"Tower"}',
                     '{"action":"final","answer":"PUBLIC_CITY","supporting_facts":[["Tower",0]]}']
        result, model = self.run_fixture('hotpotqa', environment_factory(config, self.root / 'worker', training=True),
                                         'q1', responses)
        self.assertIn('supporting_facts', model.prompts[0])
        self.assertNotIn('PUBLIC_CITY', model.prompts[0])
        self.assertIn('PUBLIC_CITY', model.prompts[1])
        self.assertIn('PUBLIC_CITY', model.prompts[2])
        self.assertNotIn('OMITTED_BY_TOP_K', model.prompts[1])
        self.assertIn('PUBLIC_B', model.prompts[1])
        self.assertIn('PUBLIC_C', model.prompts[1])
        self.assertTrue(all(t.action_valid for t in result.trajectories[0].transitions))
        grade = json.loads(next((self.root / 'worker').rglob('grade.json')).read_text())
        self.assertEqual(grade['metrics']['sp_f1'], 1.)
        self.assertEqual(result.trajectories[0].total_reward, grade['metrics']['joint_f1'])
        self.assertEqual(result.trajectories[0].cost.model_calls, 3)

    def test_fresh_initial_and_final_state_bind_same_h0_and_resume_does_not_reset(self):
        checkpoint = self.root / 'checkpoint'; checkpoint.mkdir()
        for name, seed in self.seeds.items():
            _, manifest = make_catalog_manifest(name, 'fixture', records(1, 'test'), records(150), cycles=1)
            class Proposer:
                def propose(self, request): return [{'invalid': 'fixture'}] * request.count
            class Runner:
                def rollout(self, model, harness, tasks, *, seeds, output, training=False):
                    return RolloutResult((), tuple(EpisodeResult(t, s, 0., Cost()) for t in tasks for s in seeds))
            components = Components(Proposer(), Runner(), None)
            config = LoopConfig(cycles=1, total_train_steps=1, seeds=(0,))
            output = self.root / name
            run_outer_loop(components, manifest, str(checkpoint), output, config, initial_harness=seed)
            initial = load_accepted_state(output / 'initial_agent.json', manifest)
            args = Namespace(state=output / 'deployment.json', checkpoint=None, baseline=False, protocol=None)
            final = evaluation_agent(args, manifest)
            self.assertEqual(initial.harness.to_dict(), final.harness.to_dict())
            self.assertEqual(initial.protocol['initial_seed']['revision_hash'], seed.version)
            # Execute precisely the state-loaded revision; no workspace fallback.
            class PublicEnvironment:
                def reset(self, task, seed): return 'Public fixture'
                def step(self, action): return EnvironmentStep('done', 1., True, 1.)
                def close(self): pass
            self.run_fixture(name, PublicEnvironment, 'fixture', ['public action'],
                             harness=final.harness, suffix=name + '-state-rollout')
            evolved = self.store.snapshot({**seed.files(), 'prompts/retained.txt': 'retained improvement'})
            resumable = replace(initial, harness=evolved)
            # A resumed accepted revision is allowed to differ from registered H0.
            with patch('internalization.core.seed_harnesses.bind_initial_seed', side_effect=AssertionError('H0 reloaded')):
                resumed = run_outer_loop(components, manifest, str(checkpoint), self.root / (name + '-resume'),
                                         config, accepted_state=resumable)
            self.assertEqual(resumed['harness_revision']['version'], evolved.version)
            self.assertEqual(resumed['protocol']['initial_seed']['revision_hash'], seed.version)


if __name__ == '__main__': unittest.main()
