"""Batch behavior-policy module scoring; legacy tensor bridge kept separately."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from ..harness.module import ControlModule, Harness
from ..core.types import Cost, Journal, State, digest
from ..harness.runtime import TeacherHarness, distillation_selected


@dataclass(frozen=True)
class ModuleSignal:
    selected: bool
    log_probs: tuple[float, ...]
    teacher_snapshot: str
    state_fingerprint: str


class ModuleTeacherScorer:
    """Score the rollout tokens under H+ using their own behavior policy."""
    def __init__(self, teacher, harness, target, allowed_tasks, *, mode="targeted", journal=None):
        self.teacher, self.harness, self.target = teacher, harness, target
        self.allowed_tasks, self.mode, self.journal = set(allowed_tasks), mode, journal
        self.snapshot_id = teacher.snapshot_id
        harness.without(target)

    def score(self, trajectories):
        if self.teacher.snapshot_id != self.snapshot_id: raise RuntimeError("Scorer changed inside batch")
        if hasattr(self.teacher, "assert_frozen"): self.teacher.assert_frozen()
        runtime = TeacherHarness(self.teacher, self.harness)
        signals, cost = {}, Cost()
        for trajectory in trajectories:
            if trajectory.model_version != self.snapshot_id:
                raise ValueError("H+ scorer must match the rollout behavior-policy snapshot")
            if trajectory.task_id not in self.allowed_tasks: raise ValueError("Teacher state outside train split")
            for step in trajectory.transitions:
                if not step.student_prompt.startswith(step.state.public_history):
                    raise ValueError("Student cannot omit shared public history")
                if tuple(self.teacher.prompt_ids(step.student_prompt)) != step.prompt_ids:
                    raise ValueError("Teacher/student prompt tokenization mismatch")
                if step.state.fingerprint in signals: raise ValueError("Duplicate student state in batch")
                guidance = runtime.advise(step.state)
                cost += guidance.cost
                selected = distillation_selected(guidance, self.target, self.mode)
                lp = ()
                if selected:
                    lp, used = self.teacher.score(guidance.teacher_prompt, list(step.response_ids))
                    cost += used
                    if len(lp) != len(step.response_ids) or any(not math.isfinite(x) or x > 1e-5 for x in lp):
                        raise ValueError("Invalid teacher action-token scores")
                signal = ModuleSignal(selected, tuple(lp), self.snapshot_id, step.state.fingerprint)
                signals[step.state.fingerprint] = signal
                if self.journal:
                    self.journal.append("teacher_state", state=asdict(step.state), student_prompt=step.student_prompt,
                                        guidance=asdict(guidance), signal=asdict(signal))
        if self.teacher.snapshot_id != self.snapshot_id: raise RuntimeError("Scorer changed inside batch")
        if hasattr(self.teacher, "assert_frozen"): self.teacher.assert_frozen()
        return signals, cost


def real_task_ids(infos):
    result = []
    for info in infos:
        task_id = info.get("hi_task_id") or info.get("extra.gamefile")
        if not task_id:
            raise ValueError("Environment must export hi_task_id or extra.gamefile; placeholder IDs are forbidden")
        result.append(str(task_id))
    return result


def verify_response_mask(mask):
    if any(x not in (0, 1) for x in mask):
        raise ValueError("Response mask must be binary")
    values = [int(x) for x in mask]
    if any(x not in (0, 1) for x in values):
        raise ValueError("Response mask must be binary")
    if values != sorted(values, reverse=True):
        raise ValueError("Expected contiguous student response tokens followed by padding")
    return sum(values)



class TensorTeacherScorer:
    def __init__(self, phase_path, backend):
        self.path = Path(phase_path).resolve(strict=True)
        self.phase = json.loads(self.path.read_text())
        self.phase_hash = digest(self.phase)
        self.harness = Harness(tuple(ControlModule.load(Path(p)) for p in self.phase["full_modules"]))
        if self.harness.version != self.phase["full_harness_hash"]:
            raise ValueError("Harness differs from frozen phase")
        self.target = self.phase["target"]
        self.harness.without(self.target)
        self.allowed_tasks = set(self.phase["train_task_ids"])
        if not self.allowed_tasks: raise ValueError("Explicit train task allowlist required")
        self.backend = backend
        self.journal = Journal(Path(self.phase["trace_path"]))

    def prepare(self, trainer, batch, metrics, teacher_enabled):
        import torch
        if digest(json.loads(self.path.read_text())) != self.phase_hash:
            raise RuntimeError("Phase manifest changed while training")
        self.backend.assert_frozen()
        if not teacher_enabled:
            raise RuntimeError("Harness experiment cannot silently run with teacher supervision disabled")
        required = {"obs_text", "hi_task_id", "traj_uid", "step_num"}
        if not required.issubset(batch.non_tensor_batch):
            raise ValueError(f"Missing rollout metadata: {required - batch.non_tensor_batch.keys()}")
        if "multi_modal_inputs" in batch.non_tensor_batch:
            raise ValueError("This phase supports text-only observations")
        responses = batch.batch["responses"]
        mask = batch.batch["attention_mask"][:, -responses.shape[1]:]
        log_probs = torch.zeros_like(responses, dtype=torch.float32)
        selected = torch.zeros(len(batch), dtype=torch.bool, device=responses.device)
        teacher = TeacherHarness(self.backend, self.harness)
        total_cost = Cost()
        # Legacy tensor callers supply complete decision rows. Sorting
        # preserves step order without showing future steps to the modules.
        order = sorted(range(len(batch)), key=lambda i: (
            str(batch.non_tensor_batch["traj_uid"][i]), int(batch.non_tensor_batch["step_num"][i])))
        for i in order:
            task_id = str(batch.non_tensor_batch["hi_task_id"][i])
            if task_id not in self.allowed_tasks:
                raise ValueError(f"Actual rollout task is outside train manifest: {task_id}")
            prompt = str(batch.non_tensor_batch["obs_text"][i])
            # Exact chat-template token equality catches invisible history,
            # truncation and template mismatch, beyond comparing strings.
            expected_ids = self.backend.prompt_ids(prompt)
            original_ids = batch.batch["prompts"][i]
            original_mask = batch.batch["attention_mask"][i, :-responses.shape[1]]
            if original_ids[original_mask.bool()].tolist() != expected_ids:
                raise ValueError("Teacher public prompt differs from actual student prompt tokens")
            base_prompt = str(batch.non_tensor_batch.get("obs_text_base", batch.non_tensor_batch["obs_text"])[i])
            if not prompt.startswith(base_prompt):
                raise ValueError("Student retained guidance must preserve the full common public prompt")
            state = State(task_id, str(batch.non_tensor_batch["traj_uid"][i]),
                          int(batch.non_tensor_batch["step_num"][i]), base_prompt)
            guidance = teacher.advise(state)
            total_cost += guidance.cost
            use = distillation_selected(guidance, self.target, self.phase.get("supervision", "targeted"))
            count = verify_response_mask(mask[i].tolist())
            if use and count:
                lp, score_cost = self.backend.score(guidance.teacher_prompt, responses[i, :count].tolist())
                if len(lp) != count or any(not math.isfinite(x) or x > 1e-5 for x in lp):
                    raise ValueError("Invalid teacher token log probabilities")
                log_probs[i, :count] = torch.tensor(lp, device=responses.device)
                total_cost += score_cost
                selected[i] = True
            self.journal.append("teacher_state", state=asdict(state), student_prompt=prompt, guidance=asdict(guidance),
                selected=bool(selected[i]), response_ids=responses[i, :count].tolist(),
                teacher_log_probs=log_probs[i, :count].tolist() if use else [])
        # Legacy tensor field aliases retained solely for original project tests. No teacher text or tool
        # tokens are appended to the student's actual trainable response.
        batch.batch["teacher_log_prob"] = log_probs
        batch.batch["step_teacher_log_prob"] = log_probs.clone()
        batch.batch["step_skill_mask"] = selected
        batch.batch["teacher_signal_mask"] = selected.clone()
        metrics["hi/selected_steps"] = float(selected.sum())
        metrics.update({f"hi/teacher_{k}": v for k, v in asdict(total_cost).items()})
        self.journal.append("teacher_batch_cost", cost=asdict(total_cost), phase=self.phase_hash)
        return batch
