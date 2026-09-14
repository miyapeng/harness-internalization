"""The only backend contracts consumed by the outer experiment loop."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Callable

from .trajectory import RolloutResult, Trajectory
from .types import EpisodeResult
from ..harness.module import Harness
from ..harness.revision import HarnessRevision, InternalizationTarget


@dataclass(frozen=True)
class ProposalRequest:
    checkpoint: str
    harness: Harness | HarnessRevision
    tasks: tuple[str, ...]
    trajectories: tuple[Trajectory, ...]
    scores: tuple[EpisodeResult, ...]
    history: tuple[dict, ...]
    cycle: int
    count: int
    output: Path


class ProposerBackend(Protocol):
    def propose(self, request: ProposalRequest): ...


class TaskRunner(Protocol):
    def rollout(self, model, harness: Harness | HarnessRevision, tasks: tuple[str, ...], *, seeds: tuple[int, ...],
                output: Path, training: bool = False) -> RolloutResult: ...

    def check_internalization(self, model, target: InternalizationTarget, tasks: tuple[str, ...], *, output: Path): ...


class InternalizationTargetBackend(Protocol):
    def propose_target(self, request: ProposalRequest) -> InternalizationTarget | None: ...


class TrainingBackend(Protocol):
    # The legacy teacher argument specifies a phase-frozen KL reference.
    # Module supervision must use each batch's current behavior policy instead.
    def train(self, student, teacher, h_plus: Harness | HarnessRevision, h_minus: Harness | HarnessRevision,
              trajectories: Callable[[], tuple[Trajectory, ...]] | None, *, tasks: tuple[str, ...],
              target: str | InternalizationTarget, budget: int, output: Path) -> str: ...


class RetirementEvaluator(Protocol):
    # Return independent model_decision (accept/rollback) and module_decision
    # (retire/retain). Rollback must retain the module. Legacy decision alone
    # cannot authorize accepting a newly trained checkpoint.
    def evaluate(self, before_model, after_model, h_plus: Harness | HarnessRevision, h_minus: Harness | HarnessRevision, *,
                 tasks, seeds, output, before_cells=None): ...


@dataclass
class Components:
    proposer: ProposerBackend
    runner: TaskRunner
    trainer: TrainingBackend
    retirement: RetirementEvaluator | None = None
    targets: InternalizationTargetBackend | None = None

    execution_config: dict | None = None
