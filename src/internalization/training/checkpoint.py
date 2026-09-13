"""Checkpoint persistence through model/package APIs, never a vendored merger."""
from __future__ import annotations

from pathlib import Path
from ..core.types import write_json


class CheckpointManager:
    def save(self, policy, output: Path, *, step: int, teacher_snapshot: str, behavior_snapshot=None):
        if output.exists(): raise FileExistsError(output)
        result = Path(policy.save_checkpoint(output, step=step)).resolve(strict=True)
        if not result.is_relative_to(output.resolve()) or not result.is_dir():
            raise ValueError("Checkpoint must be created inside the requested new output directory")
        write_json(result / "internalization.json", {"step": step,
            "teacher_snapshot": behavior_snapshot or teacher_snapshot, "student_snapshot": policy.snapshot_id,
            "kl_reference_snapshot": teacher_snapshot, "last_behavior_snapshot": behavior_snapshot})
        return str(result)
