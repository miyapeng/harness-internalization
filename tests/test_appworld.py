import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
import sys
from unittest.mock import patch

from internalization.benchmarks.appworld import AppWorldConfig, AppWorldEnvironment, AppWorldProcess
from internalization.benchmarks.appworld_worker import NativeAppWorld, VERSION
from internalization.benchmarks.appworld_manifest import build_manifest, aggregate_results
from internalization.core.types import Cost, EpisodeResult
from internalization.core.trajectory import read_trace_file
from internalization.harness.module import Harness, HarnessModule
from internalization.harness.runtime import Completion
from internalization.training.rollout import InteractionTaskRunner
from internalization.training.trainer import ModuleTrainer
from internalization.training import entrypoint
from internalization.command_backend import serialize_harness
import test_behavior_policy as behavior_fixtures

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("appworld_test_double", ROOT / "tests/fixtures/appworld_stub.py")
stub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stub)


class AppWorldTests(unittest.TestCase):
    def native(self, root, *, max_steps=3):
        return NativeAppWorld(stub.AppWorld, stub.load_task_ids, root, max_steps=max_steps)

    def reset(self, env, **overrides):
        return env.reset(**{"task_id":"aaaaaaa_1", "seed":4, "training":True,
                            "experiment_name":"hi_"+"a"*32, **overrides})

    def test_official_contract_public_prompt_seed_and_no_hardcoded_actions(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"APPWORLD_ROOT":directory}):
            env = self.native(directory)
            result = self.reset(env)
            self.assertEqual(env.world.executed, [])
            self.assertEqual(env.world.kwargs["random_seed"], 4)
            self.assertEqual(env.world.kwargs["ground_truth_mode"], "minimal")
            for name in ("raise_on_unsafe_syntax", "null_patch_unsafe_execution"):
                self.assertTrue(env.world.kwargs[name])
            self.assertFalse(env.world.kwargs["add_login_shortcut"])
            self.assertIn("Complete the synthetic task", result["observation"])
            self.assertNotIn("SECRET", json.dumps(result))
            world = env.world
            env.close()
            self.assertTrue(world.closed)

    def test_terminal_official_success_not_completion_flag_and_real_api_counts(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"APPWORLD_ROOT":directory}):
            env = self.native(directory)
            self.reset(env)
            first = env.step('```python\nprint("hello")\n```')
            self.assertEqual(env.world.executed, ['print("hello")'])
            self.assertEqual((first["reward"], first["done"], first["tool_calls"]), (0., False, 3))
            self.assertEqual(env.world.evaluations, 0)
            env.world.pass_tests = False
            last = env.step("apis.supervisor.complete_task()")
            self.assertTrue(last["done"])
            self.assertEqual(last["success"], 0.)
            self.assertEqual(last["tool_calls"], 2)
            self.assertEqual(env.world.evaluations, 1)
            self.assertNotIn("SECRET", json.dumps(last))
            env.close()

    def test_errors_and_step_limit_are_observations_but_grader_faults_raise(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"APPWORLD_ROOT":directory}):
            env = self.native(directory, max_steps=1)
            self.reset(env)
            result = env.step("invalid")
            self.assertFalse(result["action_valid"])
            self.assertEqual(result["termination"], "step_limit")
            self.assertEqual(env.world.evaluations, 1)
            with self.assertRaises(RuntimeError): env.step("anything")
            env.close()
            env = self.native(directory, max_steps=1)
            self.reset(env, experiment_name="hi_"+"b"*32)
            env.world.evaluate = lambda: type("BrokenGrader", (), {"num_tests":0, "total_count":0})()
            with self.assertRaisesRegex(RuntimeError, "grader"): env.step("invalid")
            env.close()

    def test_unknown_tasks_and_training_on_dev_or_test_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            env = self.native(directory)
            for task in ("bbbbbbb_1", "ccccccc_1", "ddddddd_1", "unknown"):
                with self.assertRaises(ValueError): self.reset(env, task_id=task)
            self.assertIsNone(env.world)

    def install_stub(self, root):
        packages = root / "stub_packages"
        (packages / "appworld").mkdir(parents=True)
        (packages / "appworld/__init__.py").write_text((ROOT / "tests/fixtures/appworld_stub.py").read_text())
        info = packages / f"appworld-{VERSION}.dist-info"
        info.mkdir()
        (info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: appworld\nVersion: {VERSION}\n")
        datasets = root / "data/datasets"
        datasets.mkdir(parents=True)
        for split in ("train", "dev", "test_normal", "test_challenge"):
            (datasets / (split+".txt")).write_text("\n".join(stub.load_task_ids(split)))
        import sys
        return {"HI_APPWORLD_PYTHON":sys.executable, "APPWORLD_ROOT":str(root),
                "PYTHONPATH":str(packages)+os.pathsep+os.environ.get("PYTHONPATH", "")}

    def test_real_subprocess_bridge_with_explicit_stub_package_and_repeated_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, self.install_stub(root)):
                worlds = []
                for i in range(2):
                    env = AppWorldEnvironment(AppWorldConfig(), root / f"episode-{i}", training=True)
                    try:
                        initial = env.reset("aaaaaaa_1", 5)
                        world = json.loads((env.output / "identity.json").read_text())
                        worlds.append(world["official_output_directory"])
                        step = env.step('print("hello")')
                        self.assertIn(initial, step.observation)
                        self.assertIn("hello", step.observation)
                        final = env.step("apis.supervisor.complete_task()")
                        self.assertEqual((final.reward, final.success, final.done), (1., 1., True))
                        self.assertNotIn("SECRET", final.observation)
                    finally: env.close()
                self.assertNotEqual(worlds[0], worlds[1])
                for folder in worlds: self.assertTrue(Path(folder).is_dir())

    def test_worker_error_is_not_silently_converted_into_failed_task(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, self.install_stub(root)):
                env = AppWorldEnvironment(AppWorldConfig(), root / "episode", training=True)
                try:
                    with self.assertRaisesRegex(RuntimeError, "official train"):
                        env.reset("ccccccc_1", 0)
                finally: env.close()

    def official_splits(self):
        offset, splits = 0, {}
        for split, groups in (("train",30), ("dev",19), ("test_normal",2), ("test_challenge",2)):
            splits[split] = [f"{g:07x}_{i}" for g in range(offset, offset+groups) for i in (1,2,3)]
            offset += groups
        return splits

    def test_three_cycle_manifest_preserves_official_train_and_scenario_separation(self):
        splits = self.official_splits()
        manifest = build_manifest(splits, "test-revision", versioned=False)
        self.assertEqual({k:len(v) for k,v in manifest.partitions.items()},
            {"train":27,"search":15,"dev":15,"retirement_0":30,"retirement_1":30,"retirement_2":30,
             "test_normal":6,"test_challenge":6})
        self.assertEqual(manifest.fingerprint, build_manifest(splits, "test-revision", versioned=False).fingerprint)
        self.assertTrue(set(manifest.partitions["train"]) <= set(splits["train"]))
        self.assertTrue(set(manifest.partitions["search"]) <= set(splits["train"]))
        families = {}
        for partition, tasks in manifest.partitions.items():
            for task in tasks:
                scenario = task.rsplit("_",1)[0]
                self.assertEqual(families.setdefault(scenario,partition), partition)
        with self.assertRaises(ValueError): manifest.partition("test_normal")

    def test_manifest_cannot_borrow_test_tasks_reuse_cohorts_or_split_variants(self):
        splits = self.official_splits()
        with self.assertRaises(ValueError): build_manifest(splits, "v1", cycles=5)
        with self.assertRaises(ValueError): build_manifest(splits, "v1", cohort_size=15)
        splits["train"].pop()
        with self.assertRaises(ValueError): build_manifest(splits, "v1")

    def test_task_and_scenario_metrics_and_incomplete_scenarios(self):
        rows = [EpisodeResult(f"aaaaaaa_{i}",seed,float(i!=3),Cost()) for i in (1,2,3) for seed in (0,1)]
        summary = aggregate_results(rows)
        self.assertAlmostEqual(summary["task_goal_completion"], 200/3)
        self.assertEqual(summary["scenario_goal_completion"], 0.)
        self.assertIsNone(aggregate_results(rows[:4])["scenario_goal_completion"])
        with self.assertRaises(ValueError): aggregate_results(rows[:-1])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch required")
    def test_appworld_rollout_module_scoring_and_two_real_cpu_mock_updates(self):
        from uuid import uuid4
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, self.install_stub(root)):
                factory = lambda: AppWorldEnvironment(AppWorldConfig(max_steps=2), root / uuid4().hex, training=True)
                runner = InteractionTaskRunner(factory, max_steps=2)
                fixture = behavior_fixtures.BehaviorPolicyTests()
                policy = fixture.policy()
                generate = policy.generate
                def python_action(prompt, *, purpose):
                    completion = generate(prompt, purpose=purpose)
                    if purpose == "rollout_action":
                        return Completion("apis.supervisor.complete_task()", completion.cost, completion.response_ids)
                    return completion
                policy.generate = python_action
                harness = Harness((HarnessModule.from_source(behavior_fixtures.SOURCE),))
                trainer = ModuleTrainer(runner, tasks_per_batch=1, rollouts_per_task=2)
                checkpoint = trainer.train(policy, fixture.reference(), harness, Harness(), target="target",
                    tasks=("aaaaaaa_1",), budget=2, output=root / "training")
                self.assertEqual(policy.updates, 2)
                self.assertTrue((Path(checkpoint) / "weights.pt").is_file())
                for batch in range(2):
                    trajectories = read_trace_file(root / f"training/rollout_{batch:05d}/trajectories.jsonl")
                    self.assertEqual(len(trajectories), 2)
                    for trajectory in trajectories:
                        self.assertEqual(trajectory.success, 1.)
                        self.assertEqual(trajectory.model_version, f"policy-{batch}")
                        self.assertEqual(trajectory.cost.tool_calls, 2)
                        self.assertNotIn("GUIDANCE", trajectory.transitions[0].student_prompt)
                        self.assertNotIn("SECRET", trajectory.transitions[0].state.public_history)
                        self.assertEqual(trajectory.transitions[0].response_ids, (3,4))
                for batch in policy.batches:
                    self.assertEqual(batch["response_mask"].tolist(), [[1,1],[1,1]])
                    self.assertGreater(float(batch["advantages"].mean()), 0.)

    def test_benchmark_config_is_real_dispatch_and_context_limits_are_explicit(self):
        config = AppWorldConfig.load(ROOT / "configs/appworld.json")
        self.assertGreater(config.max_prompt_tokens, 4096)
        backend = json.loads((ROOT / "configs/appworld_backend.json").read_text())
        for stage in ("train", "evaluate"):
            self.assertEqual(backend[stage][backend[stage].index("--benchmark")+1], "appworld")
        with self.assertRaises(ValueError): AppWorldConfig(rpc_timeout_s=50)
        with self.assertRaises(ValueError): AppWorldConfig(max_context=4096)

    def test_evaluation_entrypoint_routes_to_appworld_and_records_aggregate(self):
        class Model:
            snapshot_id = "eval-snapshot"
            def generate(self, prompt, *, purpose):
                return Completion("apis.supervisor.complete_task()", Cost(20,3,1), (1,2,3))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, self.install_stub(root)):
                request, response = root / "request.json", root / "evaluation/response.json"
                request.write_text(json.dumps({"stage":"evaluate", "checkpoint":"model", "harness":serialize_harness(Harness()),
                    "task_ids":stub.load_task_ids("dev"), "seeds":[0]}))
                with patch.object(sys, "argv", ["worker", "--request",str(request),"--response",str(response),
                        "--benchmark","appworld","--device","cpu"]), \
                     patch.object(entrypoint, "FrozenHFBackend", return_value=Model()) as loader:
                    entrypoint.main("evaluate")
                result = json.loads(response.read_text())
                self.assertEqual(result["benchmark_metrics"]["task_goal_completion"], 100.)
                self.assertEqual(result["benchmark_metrics"]["scenario_goal_completion"], 100.)
                self.assertEqual(result["cost"]["tool_calls"], 6)
                self.assertEqual(loader.call_args.kwargs["max_context"], 32768)
                self.assertTrue((response.parent / "benchmark_protocol.json").is_file())

    @unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch required")
    def test_training_entrypoint_routes_appworld_through_original_module_trainer(self):
        from internalization.training import verl_backend
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.dict(os.environ, self.install_stub(root)):
                policy = behavior_fixtures.BehaviorPolicyTests().policy()
                original = policy.generate
                def generate(prompt, *, purpose):
                    result = original(prompt,purpose=purpose)
                    return Completion("apis.supervisor.complete_task()", result.cost, result.response_ids) if purpose == "rollout_action" else result
                policy.generate = generate
                reference = behavior_fixtures.BehaviorPolicyTests().reference()
                full = Harness((HarnessModule.from_source(behavior_fixtures.SOURCE),))
                request, response = root / "request.json", root / "training/response.json"
                request.write_text(json.dumps({"stage":"train", "student_checkpoint":"model", "teacher_checkpoint":"model",
                    "full_harness":serialize_harness(full), "reduced_harness":serialize_harness(Harness()),
                    "target":"target", "task_ids":["aaaaaaa_1"], "optimizer_steps":1}))
                with patch.object(sys,"argv",["worker","--request",str(request),"--response",str(response),
                        "--benchmark","appworld","--device","cpu"]), \
                     patch.object(entrypoint,"FrozenHFBackend",return_value=reference), \
                     patch.object(verl_backend,"VerlPolicy",return_value=policy) as loader:
                    entrypoint.main("train")
                result = json.loads(response.read_text())
                self.assertEqual(result["attempted_update_batches"],1)
                self.assertEqual(loader.call_args.kwargs["max_prompt_tokens"],28672)
                self.assertTrue((Path(result["checkpoint"]) / "weights.pt").is_file())


if __name__ == "__main__": unittest.main()
