import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from internalization.core.interfaces import ProposalRequest
from internalization.core.types import Cost
from internalization.core.trajectory import RolloutResult
from internalization.evolution.archive import CandidateArchive
from internalization.evolution.candidate import Candidate
from internalization.evolution.proposer import APIProposer
from internalization.harness.module import Harness, HarnessModule
from internalization.harness.runtime import Completion
from internalization.training.rollout import InteractionTaskRunner, EnvironmentStep

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'NAME="test"\nKIND="planner"\nINSTRUCTION="guide"\nPERSISTENCE=0\ndef trigger(history, step):\n    return True\n'


class MigrationTests(unittest.TestCase):
    def test_original_twenty_tests_were_not_edited(self):
        hashes = json.loads((ROOT / "docs/history/migration-baseline/original-tests.sha256.json").read_text())
        for file, expected in hashes.items():
            self.assertEqual(hashlib.sha256((ROOT / file).read_bytes()).hexdigest(), expected, file)

    def test_proposer_can_be_mocked_without_any_model_or_upstream(self):
        calls = []
        def transport(payload):
            calls.append(payload)
            return {"choices": [{"message": {"content": json.dumps({"candidate_sources": [SOURCE, SOURCE]})}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        with tempfile.TemporaryDirectory() as directory:
            proposal = ProposalRequest("model", Harness(), (), (), (),
                ({"source": "prior", "search_gain": {"mean": 0.1}},), 0, 2, Path(directory))
            proposer = APIProposer(model="mock", transport=transport)
            candidates = proposer.propose(proposal)
            self.assertEqual(len(candidates), 2)
            self.assertEqual(candidates[0].parent, Harness().version)
            self.assertIn("prior", calls[0]["messages"][1]["content"])
            self.assertEqual(proposer.last_cost.input_tokens, 10)

    def test_archive_keeps_failed_and_successful_candidates_without_dev_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            archive = CandidateArchive(Path(directory) / "archive.jsonl")
            for index, status in enumerate(("invalid", "eligible")):
                archive.record(Candidate(SOURCE, "parent", 0, index), status=status,
                               search_gain={"mean": 0.2}, dev_gain={"mean": 0.9})
            self.assertEqual(len(archive.history()), 2)
            self.assertEqual([r["status"] for r in archive.history()], ["invalid", "eligible"])
            self.assertNotIn("dev_gain", json.dumps(archive.proposer_history()))

    def test_outer_imports_only_project_modules_and_standard_library(self):
        import ast
        tree = ast.parse((ROOT / "src/internalization/outer_loop.py").read_text())
        external = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and not n.level]
        self.assertEqual(set(external), {"__future__", "dataclasses", "pathlib"})

    def test_benchmark_identity_contracts(self):
        from internalization.benchmarks.webshop import split_for, check_partition
        from internalization.benchmarks.search_qa import task_id
        self.assertEqual([split_for(i) for i in (499, 500, 1499, 1500)], ["test", "dev", "dev", "train"])
        with self.assertRaises(ValueError): check_partition([500], "train")
        with self.assertRaises(ValueError): task_id({"question": "placeholder"})

    def test_alfworld_prompts_match_original_manager_and_history(self):
        from internalization.benchmarks.alfworld import AlfworldPrompt
        fixture = json.loads((ROOT / "tests/fixtures/alfworld_prompt_reference.json").read_text())
        prompt = AlfworldPrompt()
        actual = [prompt.reset(fixture["observations"][0], fixture["admissible"])]
        for action, observation in zip(fixture["actions"], fixture["observations"][1:]):
            actual.append(prompt.advance(action, observation, fixture["admissible"]))
        self.assertEqual(actual, fixture["expected"])

    @unittest.skipUnless(importlib.util.find_spec("yaml"), "PyYAML needed for external environment adapter")
    def test_external_alfworld_adapter_preserves_task_reward_and_actions(self):
        from internalization.benchmarks.alfworld import AlfworldEnvironment
        with tempfile.TemporaryDirectory() as directory:
            task = Path(directory) / "game.tw-pddl"
            task.touch()
            class Native:
                def seed(self, value): self.seed_value = value
                def reset(self):
                    return ["Room. Your task is to: look."], {"extra.gamefile": [str(task)],
                                                           "admissible_commands": [["look", "help"]]}
                def step(self, actions):
                    self.actions = actions
                    return ["Finished"], [999], [True], {"won": [True], "admissible_commands": [["look"]]}
                def close(self): self.closed = True
            native = Native()
            class ExternalPackage:
                def __init__(self, config, train_eval):
                    self.config, self.train_eval = config, train_eval
                def init_env(self, batch_size):
                    if self.game_files != [str(task)] or self.num_games != 1:
                        raise AssertionError("Task allowlist lost")
                    return native
            env = AlfworldEnvironment(ROOT / "configs/alfworld.yaml", environment_class=ExternalPackage)
            env.reset(str(task), 17)
            outcome = env.step("<think>Check room</think><action>LOOK</action>")
            self.assertEqual(native.seed_value, 17)
            self.assertEqual(native.actions, ["look"])
            self.assertEqual(outcome.reward, 10.0)
            self.assertTrue(outcome.action_valid)
            env.close()
            self.assertTrue(native.closed)

    def test_checkpoint_refuses_existing_and_escaped_outputs(self):
        from internalization.training.checkpoint import CheckpointManager
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            class Policy:
                def save_checkpoint(self, output, *, step): return root
            manager = CheckpointManager()
            with self.assertRaises(FileExistsError):
                manager.save(Policy(), root, step=1, teacher_snapshot="frozen")
            with self.assertRaises(ValueError):
                manager.save(Policy(), root / "new", step=1, teacher_snapshot="frozen")


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch needed")
class TrainingMigrationTests(unittest.TestCase):
    def test_advantages_match_pre_migration_reference(self):
        import torch
        from internalization.training.module_advantage import task_advantage, module_advantage, combine_advantages, AdvantageConfig
        reference = json.loads((ROOT / "tests/fixtures/advantage_reference.json").read_text())
        for fixture in reference["fixtures"]:
            tensors = {k: torch.tensor(fixture[k], dtype=torch.float32)
                       for k in ("mask", "rewards", "old", "teacher", "selected")}
            task = task_advantage(tensors["rewards"], tensors["mask"], fixture["groups"])
            module = module_advantage(tensors["teacher"], tensors["old"], tensors["mask"], tensors["selected"],
                                      normalize=fixture["normalize"])
            result = combine_advantages(tensors["rewards"], tensors["old"], tensors["teacher"], tensors["mask"],
                tensors["selected"], fixture["groups"], AdvantageConfig(normalize_module=fixture["normalize"]))
            for output, key in ((task, "task_advantage"), (module, "module_advantage"), (result, "combined")):
                torch.testing.assert_close(output, torch.tensor(fixture[key]), rtol=1e-6, atol=1e-7)

    def test_trainer_mock_updates_parameters_and_preserves_frozen_teacher(self):
        import torch
        from internalization.training.trainer import ModuleTrainer
        class Environment:
            def reset(self, task, seed): return "public state"
            def step(self, action):
                self.assert_action = action
                return EnvironmentStep("done", 1.0, True, 1.0)
            def close(self): pass
        class Teacher:
            snapshot_id = "frozen-teacher"
            def prompt_ids(self, prompt): return [1, 2]
            def generate(self, prompt, *, purpose): raise AssertionError("KL reference cannot generate module advice")
            def score(self, prompt, ids): return [-0.1] * len(ids), Cost(3, 0, 1, 1)
            def assert_frozen(self): pass
        class Policy:
            pad_token_id = 0
            def __init__(self):
                self.weight = torch.nn.Parameter(torch.tensor(-1.0))
                self.optimizer = torch.optim.SGD([self.weight], lr=0.1)
                self.snapshot_id = "student-0"
                self.calls = 0
            def prompt_ids(self, prompt): return [1, 2]
            def generate(self, prompt, *, purpose):
                if purpose == "planner": return Completion("PRIVATE ADVICE", Cost(2, 1, 1, 1), (1,))
                if "PRIVATE ADVICE" in prompt: raise AssertionError("Target leaked into student rollout")
                return Completion("action", Cost(2, 1, 1), (1,))
            def score(self, prompt, ids):
                # Same current policy; only module conditioning changes the score.
                lp = float(self.weight.detach()) + (0.5 if "PRIVATE ADVICE" in prompt else 0.0)
                return [lp] * len(ids), Cost(3, 0, 1)
            def update(self, batch):
                self.optimizer.zero_grad()
                loss = -(self.weight * batch.tensor_data["advantages"]).mean()
                loss.backward()
                self.optimizer.step()
                self.calls += 1
                self.snapshot_id = f"student-{self.calls}"
                return {"mock_loss": float(loss.detach())}
            def save_checkpoint(self, output, *, step):
                output.mkdir(parents=True)
                torch.save(self.weight.detach(), output / "weights.pt")
                return output
        with tempfile.TemporaryDirectory() as directory:
            student, teacher = Policy(), Teacher()
            full = Harness((HarnessModule.from_source(SOURCE),))
            trainer = ModuleTrainer(InteractionTaskRunner(Environment), tasks_per_batch=1, rollouts_per_task=2)
            checkpoint = trainer.train(student, teacher, full, Harness(), target="test", tasks=("task",),
                                       budget=2, output=Path(directory) / "train")
            self.assertGreater(float(student.weight.detach()), -1.0)
            self.assertEqual(student.calls, 2)
            self.assertEqual(teacher.snapshot_id, "frozen-teacher")
            self.assertTrue((Path(checkpoint) / "internalization.json").exists())
            from internalization.core.trajectory import read_trace_file
            restored = read_trace_file(Path(directory) / "train/rollout_00000/trajectories.jsonl")
            self.assertEqual([t.seed for t in restored], [0, 0])
            self.assertNotEqual(restored[0].episode_id, restored[1].episode_id)
            self.assertEqual(restored[0].model_version, "student-0")
            self.assertEqual(restored[0].transitions[0].response_ids, (1,))
            self.assertGreater(restored[0].cost.input_tokens, 0)

    def test_trainer_rejects_off_policy_supplier_before_update(self):
        from types import SimpleNamespace
        from internalization.core.trajectory import Trajectory
        from internalization.training.trainer import ModuleTrainer
        full = Harness((HarnessModule.from_source(SOURCE),))
        reduced = Harness()
        stale = Trajectory("task", "episode", 0, "old", reduced.version, (), 0, Cost())
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Stale/off-policy"):
                ModuleTrainer(None).train(SimpleNamespace(snapshot_id="current"),
                    SimpleNamespace(snapshot_id="teacher"), full, reduced, lambda: (stale,),
                    target="test", tasks=("task",), budget=1, output=Path(directory))

    def test_module_advantage_is_detached_and_padding_zero(self):
        import torch
        from internalization.training.module_advantage import module_advantage
        teacher = torch.tensor([[-0.2, -0.4]], requires_grad=True)
        result = module_advantage(teacher, torch.tensor([[-1., -2.]]), torch.tensor([[1, 0]]), torch.tensor([1]))
        self.assertFalse(result.requires_grad)
        self.assertEqual(result[0, 1].item(), 0)


if __name__ == "__main__": unittest.main()
