"""Project training orchestration; optimizer execution is injected or external veRL."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

from ..core.types import Cost, Journal, write_json
from .module_advantage import AdvantageConfig, combine_advantages
from .teacher_scoring import ModuleTeacherScorer
from .checkpoint import CheckpointManager
from .behavior_policy import BehaviorPolicySnapshot


@dataclass
class UpdateBatch:
    tensor_data: dict
    metadata: dict


def build_update_batch(trajectories, signals, policy, config):
    import torch
    steps = [step for trajectory in trajectories for step in trajectory.transitions]
    if not steps: raise ValueError("No student states to update")
    prompt_ids = [policy.prompt_ids(step.student_prompt) for step in steps]
    plen = max(map(len, prompt_ids))
    rlen = max(len(step.response_ids) for step in steps)
    if not plen or not rlen: raise ValueError("Empty prompt or response")
    n = len(steps)
    prompts = torch.full((n, plen), policy.pad_token_id, dtype=torch.long)
    responses = torch.full((n, rlen), policy.pad_token_id, dtype=torch.long)
    mask = torch.zeros((n, plen + rlen), dtype=torch.long)
    rewards = torch.zeros((n, rlen))
    old, teacher = torch.zeros_like(rewards), torch.zeros_like(rewards)
    selected = torch.zeros(n, dtype=torch.bool)
    groups, row = [], 0
    for trajectory in trajectories:
        for step in trajectory.transitions:
            ids, length = prompt_ids[row], len(step.response_ids)
            if not length or len(step.old_log_probs) != length: raise ValueError("Missing on-policy token probabilities")
            prompts[row, -len(ids):] = torch.tensor(ids)
            responses[row, :length] = torch.tensor(step.response_ids)
            mask[row, plen-len(ids):plen+length] = 1
            old[row, :length] = torch.tensor(step.old_log_probs)
            # Original episode reward is repeated at every decision, followed
            # by that decision's invalid-action penalty, before normalization.
            rewards[row, length-1] = trajectory.total_reward - config.invalid_action_penalty * (not step.action_valid)
            signal = signals[step.state.fingerprint]
            if signal.teacher_snapshot != trajectory.model_version:
                raise ValueError("H+ scorer and rollout behavior-policy snapshots differ")
            selected[row] = signal.selected
            if signal.selected: teacher[row, :length] = torch.tensor(signal.log_probs)
            groups.append(trajectory.group_id or trajectory.task_id)
            row += 1
    response_mask = mask[:, plen:]
    advantage = combine_advantages(rewards, old, teacher, response_mask, selected, groups, config)
    tensors = {"prompts": prompts, "responses": responses, "input_ids": torch.cat((prompts, responses), dim=-1),
        "attention_mask": mask, "position_ids": (mask.cumsum(-1)-1).clamp_min(0), "response_mask": response_mask,
        "old_log_probs": old, "advantages": advantage, "token_level_rewards": rewards,
        "module_log_probs": teacher, "module_mask": selected}
    return UpdateBatch(tensors, {"temperature": 1.0, "multi_turn": False,
                                 "behavior_snapshot": policy.snapshot_id, "task_groups":groups,
                                 "student_prompts": [step.student_prompt for step in steps]})


class ModuleTrainer:
    """Batch-aligned self-distillation; legacy teacher argument is the KL reference."""
    def __init__(self, runner, policy_loader=None, teacher_loader=None, *, config=AdvantageConfig(),
                 tasks_per_batch=4, rollouts_per_task=2, supervision="targeted", checkpoint_manager=None,
                 sampling_state=None, sampling_seed=None, environment_seed=0):
        self.sampling_state, self.sampling_seed = sampling_state, sampling_seed
        self.environment_seed = environment_seed
        self.runner, self.policy_loader, self.teacher_loader = runner, policy_loader, teacher_loader
        self.config, self.tasks_per_batch, self.rollouts_per_task = config, tasks_per_batch, rollouts_per_task
        if supervision not in ("all", "targeted"): raise ValueError("Unknown supervision mode")
        self.supervision = supervision
        self.checkpoints = checkpoint_manager or CheckpointManager()

    def train(self, student, teacher, h_plus, h_minus, trajectories=None, *, tasks, target, budget, output):
        if h_plus.without(target).version != h_minus.version:
            raise ValueError("H- must remove exactly the target module")
        if budget < 1 or not tasks: raise ValueError("Positive training budget and task allowlist required")
        queue = None
        if self.sampling_seed is not None:
            from ..core.sampling import TaskQueue
            queue = TaskQueue(tasks,self.sampling_seed,self.sampling_state)
            if len(tasks)<self.tasks_per_batch: raise ValueError("Batch requires distinct training tasks")
        def persist_sampler():
            if queue is not None:
                if self.sampling_state is not None:
                    self.sampling_state.clear(); self.sampling_state.update(queue.state())
                # Immutable per-draw record plus atomic cursor; prior batches remain auditable.
                import json, os
                snapshot=output/"sampling"/f"{queue.draws:012d}.json"
                write_json(snapshot,queue.state())
                temporary=output/"sampling_state.pending.json"
                with temporary.open("x") as stream: json.dump(queue.state(),stream,sort_keys=True)
                os.replace(temporary,output/"sampling_state.json")
        if isinstance(student, str): student = self.policy_loader(student)
        if isinstance(teacher, str): teacher = self.teacher_loader(teacher)
        if student is teacher: raise ValueError("KL reference must not alias the updating student")
        if hasattr(student, "set_reference"): student.set_reference(teacher)
        frozen_id = teacher.snapshot_id
        output.mkdir(parents=True, exist_ok=True)
        journal = Journal(output / "training.jsonl")
        total = Cost()
        updates_completed,skipped,attempted=0,0,0
        behavior_id=student.snapshot_id
        initial_optimizer_steps = getattr(student, "optimizer_steps", None)
        def summary():
            final_steps = getattr(student, "optimizer_steps", None)
            actual_steps = final_steps-initial_optimizer_steps if initial_optimizer_steps is not None and final_steps is not None else None
            return {"planned_update_batches": budget, "attempted_update_batches": attempted,
                "actor_update_calls":updates_completed, "optimizer_steps":actual_steps,
                "batches_skipped_no_student_decisions":skipped, "supervision":self.supervision,
                "cost": asdict(total), "teacher_snapshot": behavior_id, "kl_reference_snapshot": frozen_id,
                "module_scorer_refresh": "each_batch_behavior_policy", "advantage_config": asdict(self.config),
                "optimizer_algorithm":"veRL vanilla PPO (injected backend owns optimizer)",
                "outcome_estimator":"step-weighted within-task episode-outcome normalization",
                **({"sampling_state":queue.state()} if queue is not None else {})}
        try:
            for update in range(budget):
                attempted+=1
                # H- rollout and H+ scoring share the very same pre-update policy.
                # No optimizer call occurs until this read-only context has closed.
                with BehaviorPolicySnapshot(student) as behavior:
                    behavior_id = behavior.snapshot_id
                    if trajectories is None:
                        if queue is not None:
                            self.runner.sampling_batch = queue.draws//self.tasks_per_batch
                            chosen=queue.take(self.tasks_per_batch)
                            persist_sampler()  # Consumed tasks remain consumed even after rollback/failure.
                        else:
                            chosen = tuple(tasks[(update*self.tasks_per_batch+i) % len(tasks)]
                                           for i in range(min(self.tasks_per_batch, len(tasks))))
                        if queue is not None: journal.append("batch_tasks",update=update,task_ids=list(chosen),
                            rollouts_per_task=self.rollouts_per_task,environment_seed=self.environment_seed,
                            sampling_state=queue.state() if queue is not None else None)
                        samples = self.runner.rollout(behavior, h_minus, chosen, seeds=(self.environment_seed,) * self.rollouts_per_task,
                            output=output / f"rollout_{update:05d}", training=True).trajectories
                    else:
                        if not callable(trajectories): raise TypeError("Provide a fresh on-policy trajectory supplier")
                        samples = trajectories()
                    behavior.assert_frozen()
                    if any(t.model_version != behavior_id or t.harness_version != h_minus.version for t in samples):
                        raise ValueError("Stale/off-policy trajectories or wrong student harness")
                    if any(t.task_id not in tasks for t in samples): raise ValueError("Training task outside allowlist")
                    for trajectory in samples: total += trajectory.cost
                    if not any(t.transitions for t in samples):
                        # A tool-only episode still has an outcome and real costs, but no
                        # response tokens on which to attach an actor/distillation loss.
                        skipped+=1
                        journal.append("batch_skipped",update=update,reason="no_student_decisions",
                            behavior_snapshot=behavior_id,
                            episodes=[{"task_id":t.task_id,"episode_id":t.episode_id,
                                "total_reward":t.total_reward,"success":t.success,"cost":asdict(t.cost)} for t in samples])
                        continue
                    scorer = ModuleTeacherScorer(behavior, h_plus, target, set(tasks), mode=self.supervision, journal=journal)
                    signals, teacher_cost = scorer.score(samples)
                    total += teacher_cost
                    batch = build_update_batch(samples, signals, behavior, self.config)
                updates_completed+=1  # Count the call even if the optimizer subsequently fails.
                try:
                    metrics = student.update(batch)
                finally:
                    total += getattr(student, "last_update_cost", Cost())
                if student.snapshot_id == behavior_id:
                    raise RuntimeError("Student update must advance its behavior-policy snapshot ID")
                if teacher.snapshot_id != frozen_id: raise RuntimeError("KL reference changed during phase")
                if hasattr(teacher, "assert_frozen"): teacher.assert_frozen()
                journal.append("update", update=update, metrics=metrics, teacher_snapshot=behavior_id,
                    behavior_snapshot=behavior_id, student_snapshot=student.snapshot_id, kl_reference_snapshot=frozen_id)
        except Exception as exc:
            self.last_cost=total
            self.last_summary={**summary(), "status":"failed", "error":str(exc)}
            write_json(output / "training_summary.json", self.last_summary)
            raise
        self.last_cost = total
        self.last_summary = summary()
        if updates_completed == 0:
            self.last_summary["status"] = "no_actor_updates"
            write_json(output / "training_summary.json", self.last_summary)
            from ..core.execution_config import NoActorUpdates
            raise NoActorUpdates(self.last_summary)
        result = self.checkpoints.save(student, output / "checkpoint", step=updates_completed,
            teacher_snapshot=frozen_id, behavior_snapshot=behavior_id)
        self.last_summary.update(status="trained", checkpoint=result)
        write_json(output / "training_summary.json", self.last_summary)
        return result
