"""Project implementation of module-conditioned OPID/GiGPO mathematics.

No analyzer, hindsight skill retrieval or episode-skill supervision. Defaults
match the original project's enabled path, including step-weighted task stats.
Both log-prob terms must use the same batch behavior policy: teacher_log_probs
score H+ before any update; old_log_probs are cached from the H- rollout.
See THIRD_PARTY_NOTICES.md for algorithm provenance and parity fixtures.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdvantageConfig:
    module_weight: float = 0.001
    mode: str = "mean_std_norm"
    normalize_module: bool = False
    clip_module: float | None = None
    invalid_action_penalty: float = 0.1
    epsilon: float = 1e-6


def task_advantage(token_rewards, response_mask, groups, *, mode="mean_std_norm", epsilon=1e-6):
    import torch
    if mode not in ("mean_norm", "mean_std_norm"): raise ValueError("Unknown task normalization")
    if token_rewards.shape != response_mask.shape or len(groups) != token_rewards.shape[0]:
        raise ValueError("Reward/mask/group shape mismatch")
    with torch.no_grad():
        scores = token_rewards.sum(-1)
        normalized = torch.zeros_like(scores)
        for group in dict.fromkeys(groups):
            indices = [i for i, value in enumerate(groups) if value == group]
            values = scores[indices]
            # Deliberately count each step, including repeated trajectory IDs.
            center = values.mean() if len(indices) > 1 else values.new_tensor(0.0)
            scale = values.std(correction=1) if len(indices) > 1 else values.new_tensor(1.0)
            normalized[indices] = values - center
            if mode == "mean_std_norm": normalized[indices] /= scale + epsilon
        return normalized[:, None] * response_mask


def module_advantage(teacher_log_probs, old_log_probs, response_mask, selected, *,
                     normalize=False, clip=None, epsilon=1e-6):
    import torch
    if teacher_log_probs.shape != old_log_probs.shape or teacher_log_probs.shape != response_mask.shape:
        raise ValueError("Teacher/student/mask shape mismatch")
    if selected.ndim == 1: selected = selected[:, None]
    if selected.shape not in (response_mask.shape, (response_mask.shape[0], 1)):
        raise ValueError("Module selector shape mismatch")
    valid = response_mask.bool() & selected.bool()
    result = (teacher_log_probs - old_log_probs).detach() * valid
    if normalize and valid.any():
        values = result[valid]
        centered = values - values.mean()
        std = values.std(correction=0)
        result[valid] = centered / (std + epsilon) if std > epsilon else centered
    if clip is not None:
        if clip < 0: raise ValueError("Module clip must be nonnegative")
        result = result.clamp(-clip, clip)
    return result


def combine_advantages(token_rewards, old_log_probs, teacher_log_probs, response_mask,
                       selected, groups, config=AdvantageConfig()):
    task = task_advantage(token_rewards, response_mask, groups, mode=config.mode, epsilon=config.epsilon)
    module = module_advantage(teacher_log_probs, old_log_probs, response_mask, selected,
        normalize=config.normalize_module, clip=config.clip_module, epsilon=config.epsilon)
    return (task + config.module_weight * module).detach()
