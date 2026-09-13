"""Synthetic CPU integration fixture, explicitly NOT benchmark evidence.

Uses real student-state visits and tabular updates. Token counts and latency
are synthetic units; no neural model, OPID optimizer or LLM search is run.
"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .manifest import TaskManifest
from .outer_loop import LoopConfig, run_outer_loop
from .records import Cost, EpisodeResult, Journal, State, write_json
from .retirement_eval import RetirementPolicy
from .teacher_harness import Completion, TeacherHarness, distillation_selected


KINDS = ("planner", "review", "recovery")
ACTIONS = {"planner": "open", "review": "inspect", "recovery": "retry"}


class ToyModel:
    def __init__(self, table):
        self.table = dict(table)
        self.snapshot_id = "toy:" + repr(sorted(self.table.items()))

    def generate(self, prompt, *, purpose):
        stage = prompt.split("stage=", 1)[1].split()[0]
        if purpose in KINDS:
            action = ACTIONS[stage]
        elif "[Internal" in prompt:
            action = prompt.rsplit("\n", 1)[-1]
        else:
            action = self.table.get(stage, "wait")
        return Completion(action, Cost(len(prompt.split()), 1, 1, int(purpose != "action"), latency_s=1))

    def score(self, prompt, response_ids):
        raise NotImplementedError("Toy demo uses tabular action imitation, never reports OPID gradients")


class ToyBackend:
    def __init__(self):
        self.checkpoints = {"toy:initial": {}}

    def propose(self, checkpoint, harness, tasks, cycle, count, output):
        output.mkdir(parents=True)
        kind = KINDS[cycle % 3]
        paths = []
        for i in range(count):
            path = output / f"candidate_{i}.py"
            # Two deterministic fixtures, no claim of automatic code discovery.
            predicate = f'"stage={kind}" in history' if i == 0 else 'False'
            source = (f'NAME = "{kind}_{cycle}"\nKIND = "{kind}"\n'
                      f'INSTRUCTION = "Provide {kind} guidance."\nPERSISTENCE = 0\n'
                      f'def trigger(history, step):\n    return {predicate}\n')
            with path.open("x") as f: f.write(source)
            paths.append(path)
        return paths

    def evaluate(self, checkpoint, harness, tasks, seeds, output):
        journal = Journal(output / "trajectories.jsonl")
        model = ToyModel(self.checkpoints[checkpoint])
        teacher = TeacherHarness(model, harness)
        results = []
        for task in tasks:
            stage = KINDS[int(task.rsplit("-", 1)[1]) % 3]
            for seed in seeds:
                state = State(task, f"{task}:{seed}", 0, f"stage={stage} Choose a valid next action.")
                guidance = teacher.advise(state)
                action = model.generate(guidance.teacher_prompt, purpose="action")
                cost = guidance.cost + action.cost + Cost(tool_calls=1, latency_s=1)
                result = EpisodeResult(task, seed, float(action.text == ACTIONS[stage]), cost)
                results.append(result)
                journal.append("toy_transition", state=asdict(state), action=action.text,
                               success=result.success, cost=asdict(cost))
        return results

    def train(self, checkpoint, full, reduced, target, tasks, budget, output):
        journal = Journal(output / "student_states.jsonl")
        frozen = ToyModel(self.checkpoints[checkpoint])
        teacher = TeacherHarness(frozen, full)
        table = dict(self.checkpoints[checkpoint])
        # Each update starts a fresh student episode, then visits up to 3 states.
        for update in range(budget):
            task = tasks[update % len(tasks)]
            stage = KINDS[int(task.rsplit("-", 1)[1]) % 3]
            history = f"stage={stage} Choose a valid next action."
            rollout = TeacherHarness(ToyModel(table), reduced)
            for step in range(3):
                state = State(task, f"train:{update}", step, history)
                student_guidance = rollout.advise(state)
                action = rollout.model.generate(student_guidance.teacher_prompt, purpose="action")
                # Only the student's action advances the toy environment.
                success = action.text == ACTIONS[stage]
                guidance = teacher.advise(state)
                selected = distillation_selected(guidance, target)
                label = frozen.generate(guidance.teacher_prompt, purpose="action").text
                if selected: table[stage] = label
                journal.append("toy_training_step", state=asdict(state), student_action=action.text,
                               teacher_action=label, selected=selected, success=success)
                if success: break
                history += f" Previous action {action.text}: invalid."
            teacher.forget_episode(f"train:{update}")
        next_checkpoint = f"toy:checkpoint:{len(self.checkpoints)}"
        self.checkpoints[next_checkpoint] = table
        write_json(output / "checkpoint.json", {"table": table, "synthetic": True})
        return next_checkpoint


def run_demo(output: Path):
    partitions = {p: tuple(f"{p}-{i}" for i in range(60))
                  for p in ("train", "search", "dev", "retirement_0", "retirement_1", "retirement_2", "test")}
    manifest = TaskManifest("toy-only", "toy-v1", partitions)
    result = run_outer_loop(ToyBackend(), manifest, "toy:initial", output,
                LoopConfig(total_train_steps=180, seeds=(0,)),
                RetirementPolicy(bootstrap_samples=500, min_token_saving_fraction=0.0))
    write_json(output / "DISCLAIMER.json", {"synthetic": True,
        "warning": "CPU wiring test only. Deterministic candidates, tabular updates and synthetic costs. Not OPID or benchmark evidence."})
    return result
