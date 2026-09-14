"""On-policy environment interaction, independent of environment/train frameworks."""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Protocol

from ..core.types import Cost, Journal, State
from ..core.trajectory import RolloutResult, Trajectory, Transition
from ..harness.runtime import TeacherHarness


@dataclass(frozen=True)
class EnvironmentStep:
    observation: str
    reward: float
    done: bool
    success: float
    action_valid: bool = True
    tool_calls: int = 1
    # context replaces the environment view; delta appends one new event.
    observation_kind: str = "delta"

    def __post_init__(self):
        if self.observation_kind not in ("context", "delta"):
            raise ValueError("Unknown observation_kind")


class Environment(Protocol):
    def reset(self, task_id: str, seed: int) -> str: ...
    def step(self, action: str) -> EnvironmentStep: ...
    def close(self): ...


class InteractionTaskRunner:
    def __init__(self, environment_factory, model_loader=None, *, max_steps=30, code_sandbox=None, supervision="targeted"):
        self.environment_factory, self.model_loader = environment_factory, model_loader
        self.max_steps = max_steps
        self.code_sandbox = code_sandbox
        self.supervision = supervision

    def rollout(self, model, harness, tasks, *, seeds, output, training=False):
        from ..harness.revision import HarnessRevision
        if isinstance(harness,HarnessRevision):
            from .revision_rollout import rollout_revision
            return rollout_revision(self,model,harness,tasks,seeds=seeds,output=output,training=training)
        if isinstance(model, str):
            if self.model_loader is None: raise ValueError("A model loader is required for checkpoint paths")
            model = self.model_loader(model)
        journal = Journal(output / "trajectories.jsonl")
        trajectories = []
        for task in tasks:
            for replica, seed in enumerate(seeds):
                environment = self.environment_factory()
                episode = f"{task}:{seed}" + (f":replica:{replica}" if training else "")
                runtime = TeacherHarness(model, harness)
                steps, cost, success = [], Cost(), 0.0
                started = time.perf_counter()
                try:
                    observation = environment.reset(task, seed)
                    for index in range(self.max_steps):
                        state = State(task, episode, index, observation)
                        advice = runtime.advise(state)
                        action = model.generate(advice.teacher_prompt, purpose="rollout_action" if training else "action")
                        ids = tuple(action.response_ids)
                        old_lp, scoring_cost = (), Cost()
                        if training:
                            if not ids: raise ValueError("Training rollout must retain exact generated token IDs")
                            old_lp, scoring_cost = model.score(advice.teacher_prompt, list(ids))
                        outcome = environment.step(action.text)  # the sole environment mutation
                        step_cost = advice.cost + action.cost + scoring_cost + Cost(tool_calls=outcome.tool_calls)
                        cost += step_cost
                        transition = Transition(state, advice.teacher_prompt, action.text, ids, outcome.reward,
                                                outcome.done, outcome.action_valid, tuple(old_lp), step_cost,
                                                tuple(model.prompt_ids(advice.teacher_prompt)) if training else ())
                        steps.append(transition)
                        success = outcome.success
                        observation = (outcome.observation if outcome.observation_kind == "context" else
                            observation+"\n[Student action]\n"+action.text+"\n[Environment observation]\n"+outcome.observation)
                        journal.append("transition", state=asdict(state), action=action.text,
                            student_prompt=advice.teacher_prompt, response_ids=ids, reward=outcome.reward,
                            success=success, done=outcome.done, action_valid=outcome.action_valid,
                            cost=asdict(step_cost))
                        if outcome.done: break
                finally:
                    environment.close()
                fields = asdict(cost)
                fields["latency_s"] = time.perf_counter() - started
                trajectory = Trajectory(task, episode, seed, model.snapshot_id, harness.version,
                    tuple(steps), success, Cost(**fields), task)
                trajectories.append(trajectory)
                journal.append("trajectory", trajectory=asdict(trajectory))
        return RolloutResult(tuple(trajectories), tuple(t.outcome for t in trajectories))

    def check_internalization(self, model, target, tasks, *, output):
        from .revision_scoring import check_internalization
        return check_internalization(self,model,target,tasks,output=output)
