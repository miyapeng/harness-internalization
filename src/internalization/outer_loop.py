"""Experiment orchestration depending only on project interfaces and types."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from .core.adapters import components_for
from .core.manifest import TaskManifest
from .evaluation.retirement import RetirementPolicy
from .evaluation.attribution import AttributionPolicy

@dataclass(frozen=True)
class LoopConfig:
    cycles: int = 3
    candidates_per_cycle: int = 2
    total_train_steps: int = 300
    seeds: tuple[int, ...] = (0, 1, 2)

    def __post_init__(self):
        if self.cycles < 1 or self.candidates_per_cycle < 1 or self.total_train_steps < self.cycles:
            raise ValueError("Invalid loop budget")
        if self.total_train_steps % self.cycles:
            raise ValueError("Training budget must divide equally across cycles")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Unique evaluation seeds required")


def run_outer_loop(backend, manifest: TaskManifest, checkpoint: str,
                   output: Path, config=LoopConfig(), policy=RetirementPolicy(), *,
                   attribution_policy=AttributionPolicy(), initial_harness=None, accepted_state=None):
    components = components_for(backend)
    if accepted_state is not None:
        if initial_harness is not None: raise ValueError("Choose an accepted Agent or an initial Harness, not both")
        accepted_state.check_resume(manifest,config,policy,attribution_policy)
        if checkpoint != accepted_state.checkpoint: raise ValueError("Accepted checkpoint mismatch")
        from .harness.revision import HarnessRevision
        if isinstance(accepted_state.harness,HarnessRevision): initial_harness=accepted_state.harness
    if initial_harness is not None:
        from .harness.revision import HarnessRevision
        from .revision_loop import run_revision_loop
        if not isinstance(initial_harness,HarnessRevision): raise TypeError("Expected an explicit runnable HarnessRevision")
        return run_revision_loop(components,manifest,checkpoint,initial_harness,output,config,policy,attribution_policy,
                                 accepted_state=accepted_state)
    raise ValueError("Legacy module evolution is retired; supply an executable HarnessRevision")
