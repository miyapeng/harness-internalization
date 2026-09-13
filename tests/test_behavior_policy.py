"""Behavior-policy alignment regressions; CPU mocks, not a veRL/GPU run."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from internalization.core.types import Cost, State
from internalization.core.trajectory import Trajectory, Transition
from internalization.harness.module import Harness, HarnessModule
from internalization.harness.runtime import Completion
from internalization.training.behavior_policy import BehaviorPolicySnapshot
from internalization.training.rollout import EnvironmentStep, InteractionTaskRunner
from internalization.training.teacher_scoring import ModuleTeacherScorer, ModuleSignal
from internalization.training.trainer import ModuleTrainer, build_update_batch
from internalization.training.module_advantage import AdvantageConfig

SOURCE = 'NAME="target"\nKIND="planner"\nINSTRUCTION="analyze"\nPERSISTENCE=0\ndef trigger(history, step):\n    return True\n'


@unittest.skipUnless(importlib.util.find_spec("torch"), "CPU torch required")
class BehaviorPolicyTests(unittest.TestCase):
    def policy(self, *, effect=0.5):
        import torch

        class Policy:
            pad_token_id = 0
            def __init__(self):
                self.model = torch.nn.Linear(1, 1, bias=False)
                with torch.no_grad(): self.model.weight.fill_(-2.0)
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
                self.snapshot_id, self.updates = "policy-0", 0
                self.events, self.batches = [], []
            def prompt_ids(self, prompt): return [1, 2]
            def set_reference(self, reference): self.reference = reference
            def generate(self, prompt, *, purpose):
                if torch.is_grad_enabled() or self.model.training:
                    raise AssertionError("Inference must be read-only and eval")
                self.events.append(("generate", purpose, self.snapshot_id))
                if purpose == "planner": return Completion("GUIDANCE", Cost(2, 1, 1, 1))
                if "GUIDANCE" in prompt: raise AssertionError("Target advice leaked into student generation")
                return Completion("act", Cost(2, 2, 1), (3, 4))
            def score(self, prompt, ids):
                if torch.is_grad_enabled() or self.model.training:
                    raise AssertionError("Scoring must be read-only and eval")
                lp = float(self.model.weight[0, 0]) + (effect if "GUIDANCE" in prompt else 0)
                self.events.append(("score", self.snapshot_id, tuple(ids), lp, "GUIDANCE" in prompt))
                return [lp] * len(ids), Cost(4, 0, 1, 1)
            def update(self, batch):
                if not torch.is_grad_enabled(): raise AssertionError("Update is inside inference context")
                self.events.append(("update", self.snapshot_id))
                self.batches.append({k: v.clone() for k, v in batch.tensor_data.items()})
                # An independent outcome-gradient surrogate ensures parameter
                # movement even in the zero module-effect regression below.
                self.optimizer.zero_grad()
                loss = -self.model.weight.sum() - self.model.weight.sum() * batch.tensor_data["advantages"].mean()
                loss.backward()
                self.optimizer.step()
                self.updates += 1
                self.snapshot_id = f"policy-{self.updates}"
                return {"mock_loss": float(loss.detach())}
            def save_checkpoint(self, output, *, step):
                output.mkdir()
                torch.save(self.model.state_dict(), output / "weights.pt")
                return output
        return Policy()

    def reference(self):
        class Reference:
            snapshot_id = "phase-reference"
            def assert_frozen(self): pass
            def generate(self, *args, **kwargs): raise AssertionError("Phase reference used as H+ scorer")
            def score(self, *args, **kwargs): raise AssertionError("Mock optimizer does not need reference scoring")
        return Reference()

    def run_phase(self, root, policy, *, source=SOURCE, batches=3):
        class Environment:
            def reset(self, task, seed): return "PUBLIC"
            def step(self, action): return EnvironmentStep("DONE", 1.0, True, 1.0)
            def close(self): pass
        full = Harness((HarnessModule.from_source(source),))
        trainer = ModuleTrainer(InteractionTaskRunner(Environment), tasks_per_batch=1, rollouts_per_task=2)
        reference = self.reference()
        checkpoint = trainer.train(policy, reference, full, Harness(), target="target", tasks=("task",),
                                   budget=batches, output=root)
        return checkpoint, reference

    def test_three_batches_share_behavior_weights_tokens_and_update_order(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            root, policy = Path(directory), self.policy()
            checkpoint, reference = self.run_phase(root, policy)
            score_batches = []
            pending = []
            for event in policy.events:
                if event[0] == "score": pending.append(event)
                elif event[0] == "update":
                    self.assertEqual({e[1] for e in pending}, {event[1]})
                    score_batches.append(pending)
                    pending = []
            self.assertEqual(len(score_batches), 3)
            self.assertEqual([b[0][1] for b in score_batches], ["policy-0", "policy-1", "policy-2"])
            for batch in score_batches:
                self.assertEqual(len(batch), 4)  # two cached H- scores, two H+ scores
                minus, plus = batch[:2], batch[2:]
                self.assertTrue(all(e[2] == (3, 4) for e in batch))
                self.assertTrue(all(not e[4] for e in minus))
                self.assertTrue(all(e[4] for e in plus))
                self.assertAlmostEqual(plus[0][3]-minus[0][3], 0.5)
            self.assertGreater(score_batches[1][0][3], score_batches[0][0][3])
            self.assertEqual(sum(e[0] == "generate" and e[1] == "rollout_action" for e in policy.events), 6)
            self.assertFalse(any(e[0] == "generate" and e[1] == "action" for e in policy.events))
            for batch in policy.batches:
                torch.testing.assert_close(batch["advantages"], torch.full((2, 2), 0.0005))
                self.assertFalse(batch["advantages"].requires_grad)
            rows = [json.loads(line) for line in (root / "training.jsonl").read_text().splitlines()]
            updates = [r for r in rows if r["kind"] == "update"]
            self.assertEqual([r["teacher_snapshot"] for r in updates], ["policy-0", "policy-1", "policy-2"])
            self.assertTrue(all(r["teacher_snapshot"] == r["behavior_snapshot"] for r in updates))
            self.assertTrue(all(r["kl_reference_snapshot"] == reference.snapshot_id for r in updates))
            manifest = json.loads((Path(checkpoint) / "internalization.json").read_text())
            self.assertEqual(manifest["last_behavior_snapshot"], "policy-2")
            self.assertEqual(manifest["student_snapshot"], "policy-3")

    def test_no_module_effect_stays_zero_despite_policy_drift(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            policy = self.policy(effect=0)
            self.run_phase(Path(directory), policy)
            self.assertGreater(float(policy.model.weight.detach()[0, 0]), -2)
            for batch in policy.batches:
                torch.testing.assert_close(batch["module_log_probs"], batch["old_log_probs"], rtol=0, atol=0)
                torch.testing.assert_close(batch["advantages"], torch.zeros_like(batch["advantages"]), rtol=0, atol=0)

    def test_untriggered_target_has_no_teacher_term(self):
        import torch
        with tempfile.TemporaryDirectory() as directory:
            policy = self.policy()
            self.run_phase(Path(directory), policy, source=SOURCE.replace("return True", "return False"), batches=1)
            self.assertFalse(any(e[0] == "generate" and e[1] == "planner" for e in policy.events))
            self.assertFalse(policy.batches[0]["module_mask"].any())
            torch.testing.assert_close(policy.batches[0]["advantages"], torch.zeros((2, 2)))

    def test_snapshot_rejects_in_place_parameter_mutation(self):
        import torch
        policy = self.policy()
        with self.assertRaisesRegex(RuntimeError, "parameters or buffers changed"):
            with BehaviorPolicySnapshot(policy) as behavior:
                with torch.no_grad(): policy.model.weight.add_(1)
                behavior.score("PUBLIC", [3, 4])

    def test_retained_guidance_is_recomputed_by_the_same_batch_policy(self):
        policy = self.policy()
        source = SOURCE.replace('NAME="target"', 'NAME="retained"')
        reduced = Harness((HarnessModule.from_source(source),))
        full = Harness(reduced.modules + (HarnessModule.from_source(SOURCE.replace('KIND="planner"', 'KIND="recovery"')),))
        calls = []
        def generate(prompt, *, purpose):
            calls.append((policy.snapshot_id, purpose, prompt))
            if purpose == "planner": return Completion("RETAINED", Cost(2, 1, 1, 1))
            if purpose == "recovery": return Completion("TARGET", Cost(2, 1, 1, 1))
            if "TARGET" in prompt: raise AssertionError("Target entered student rollout")
            return Completion("act", Cost(2, 2, 1), (3, 4))
        policy.generate = generate
        class Environment:
            def reset(self, task, seed): return "PUBLIC"
            def step(self, action): return EnvironmentStep("DONE", 1., True, 1.)
            def close(self): pass
        with tempfile.TemporaryDirectory() as directory:
            with BehaviorPolicySnapshot(policy) as behavior:
                trajectories = InteractionTaskRunner(Environment).rollout(behavior, reduced, ("task",),
                    seeds=(0,), output=Path(directory), training=True).trajectories
                ModuleTeacherScorer(behavior, full, "target", {"task"}).score(trajectories)
        planners = [prompt for _, purpose, prompt in calls if purpose == "planner"]
        self.assertEqual(len(planners), 2)
        self.assertEqual(planners[0], planners[1])
        self.assertEqual({snapshot for snapshot, _, _ in calls}, {"policy-0"})
        self.assertIn("RETAINED", next(prompt for _, purpose, prompt in calls if purpose == "recovery"))

    def test_snapshot_expires_before_optimizer_work(self):
        policy = self.policy()
        with BehaviorPolicySnapshot(policy) as behavior:
            behavior.score("PUBLIC", [3, 4])
        with self.assertRaisesRegex(RuntimeError, "outside its batch lifetime"):
            behavior.score("PUBLIC", [3, 4])
        self.assertTrue(policy.model.training)  # original mode restored

    def test_scorer_and_batch_builder_reject_mismatched_snapshot(self):
        policy = self.policy()
        state = State("task", "e", 0, "PUBLIC")
        step = Transition(state, "PUBLIC", "act", (3, 4), old_log_probs=(-2., -2.), prompt_ids=(1, 2))
        trajectory = Trajectory("task", "e", 0, "other-policy", Harness().version, (step,), 0, Cost())
        with BehaviorPolicySnapshot(policy) as behavior:
            scorer = ModuleTeacherScorer(behavior, Harness((HarnessModule.from_source(SOURCE),)), "target", {"task"})
            with self.assertRaisesRegex(ValueError, "behavior-policy snapshot"):
                scorer.score((trajectory,))
            signals = {state.fingerprint: ModuleSignal(True, (-1., -1.), behavior.snapshot_id, state.fingerprint)}
            with self.assertRaisesRegex(ValueError, "snapshots differ"):
                build_update_batch((trajectory,), signals, behavior, AdvantageConfig())
