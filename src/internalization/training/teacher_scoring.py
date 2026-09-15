"""Batch behavior-policy scoring for an executable revision target."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleSignal:
    selected: bool
    log_probs: tuple[float, ...]
    teacher_snapshot: str
    state_fingerprint: str


class ModuleTeacherScorer:
    """Score the rollout tokens under H+ using their own behavior policy."""
    def __init__(self, teacher, harness, target, allowed_tasks, *, mode="targeted", journal=None):
        from ..harness.revision import InternalizationTarget
        if not isinstance(target,InternalizationTarget):
            raise TypeError("Training requires an executable InternalizationTarget; legacy module scoring is retired")
        self.teacher, self.harness, self.target = teacher, harness, target
        self.allowed_tasks, self.mode, self.journal = set(allowed_tasks), mode, journal
        self.snapshot_id = teacher.snapshot_id
        harness.without(target)

    def score(self, trajectories):
        from ..harness.revision import InternalizationTarget
        if isinstance(self.target,InternalizationTarget):
            from .revision_scoring import score_revision
            return score_revision(self,trajectories)
        raise TypeError("Training requires an executable InternalizationTarget; legacy module scoring is retired")
