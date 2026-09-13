from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .module import Harness
from ..core.types import Cost, State


@dataclass(frozen=True)
class Completion:
    text: str
    cost: Cost
    response_ids: tuple[int, ...] = ()


class ModelBackend(Protocol):
    snapshot_id: str
    def generate(self, prompt: str, *, purpose: str) -> Completion: ...
    def score(self, prompt: str, response_ids: list[int]) -> tuple[list[float], Cost]: ...


@dataclass(frozen=True)
class Guidance:
    state_fingerprint: str
    teacher_snapshot: str
    harness_version: str
    teacher_prompt: str
    active_modules: tuple[str, ...]
    triggered_modules: tuple[str, ...]
    cost: Cost


class TeacherHarness:
    """No environment handle or hidden trajectory is accepted by this interface.

Persistence selects later decisions but regenerates advice from their current
public input, never carrying a teacher-only memory into the next state.
"""
    def __init__(self, model: ModelBackend, harness: Harness):
        self.model, self.harness = model, harness
        self.snapshot_id, self.version = model.snapshot_id, harness.version
        self._until = {}
        self._last_step = {}

    def advise(self, state: State) -> Guidance:
        if self.model.snapshot_id != self.snapshot_id or self.harness.version != self.version:
            raise RuntimeError("Teacher or harness changed inside a training phase")
        previous = self._last_step.get(state.episode_id, -1)
        if state.step <= previous:
            raise ValueError("States must be supplied once, in episode order")
        self._last_step[state.episode_id] = state.step
        context = state.public_history
        active, triggered, cost = [], [], Cost()
        for module in self.harness.modules:
            key = state.episode_id, module.name
            if module.triggered(state):
                self._until[key] = state.step + module.persistence
                triggered.append(module.name)
            if state.step > self._until.get(key, -1):
                continue
            active.append(module.name)
            if module.kind == "review":
                draft = self.model.generate(context, purpose="draft")
                cost += draft.cost
                query = f"{context}\n\nInternal draft (unexecuted):\n{draft.text}"
            else:
                query = context
            advice = self.model.generate(f"{query}\n\n{module.instruction}", purpose=module.kind)
            cost += advice.cost
            context += f"\n\n[Internal {module.kind} guidance]\n{advice.text}"
        return Guidance(state.fingerprint, self.snapshot_id, self.version, context,
                        tuple(active), tuple(triggered), cost)

    def forget_episode(self, episode_id):
        self._last_step.pop(episode_id, None)
        self._until = {k: v for k, v in self._until.items() if k[0] != episode_id}


def distillation_selected(guidance: Guidance, target: str, mode: str = "targeted"):
    if mode not in ("targeted", "all"):
        raise ValueError("Supervision mode must be targeted or all")
    return mode == "all" or target in guidance.active_modules
