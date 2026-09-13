"""Normal external veRL 0.5.0 PPO actor adapter; no vendored training framework.

The default adapter uses veRL's single-process actor with a normal HF module.
Distributed users inject a TrainingBackend/worker factory rather than copying
cluster orchestration here. GPU execution of this adapter is not yet verified.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from .teacher_backend import FrozenHFBackend
from ..core.types import Cost


ACTOR_CONFIG = {
    "ppo_mini_batch_size": 8, "ppo_micro_batch_size_per_gpu": 1, "ppo_epochs": 1,
    "clip_ratio": 0.2, "clip_ratio_low": 0.2, "clip_ratio_high": 0.2, "clip_ratio_c": 3.0,
    "loss_agg_mode": "token-mean", "use_kl_loss": True, "kl_loss_coef": 0.01,
    "kl_loss_type": "low_var_kl", "entropy_coeff": 0.001, "grad_clip": 1.0,
    "policy_loss": {"loss_mode": "vanilla"}, "use_remove_padding": False,
    "use_dynamic_bsz": False, "use_fused_kernels": False, "use_torch_compile": False,
    "ulysses_sequence_parallel_size": 1, "entropy_from_logits_with_chunking": False,
    "entropy_checkpointing": False,
}


class VerlPolicy(FrozenHFBackend):
    def __init__(self, checkpoint, *, device="cuda:0", max_context=8192, max_action_tokens=512,
                 max_new_tokens=192, max_prompt_tokens=4096):
        # Imports occur only when this real optimizer backend is requested.
        from importlib.metadata import version
        if version("verl") != "0.5.0":
            raise RuntimeError("This adapter is pinned to external verl==0.5.0; validate migrations explicitly")
        from omegaconf import OmegaConf
        from verl.workers.actor.dp_actor import DataParallelPPOActor
        super().__init__(checkpoint, device=device, max_context=max_context,
                         max_action_tokens=max_action_tokens, max_new_tokens=max_new_tokens)
        self.max_prompt_tokens = max_prompt_tokens
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None: self.pad_token_id = self.tokenizer.eos_token_id
        self.model.requires_grad_(True)
        self.model.gradient_checkpointing_enable()
        self.optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=1e-6, weight_decay=0.01)
        if not self.torch.distributed.is_initialized():
            self.rendezvous = tempfile.TemporaryDirectory(prefix="hi-verl-")
            self.torch.distributed.init_process_group("gloo", rank=0, world_size=1,
                init_method=(Path(self.rendezvous.name) / "rendezvous").as_uri())
        elif self.torch.distributed.get_world_size() != 1:
            raise RuntimeError("Use an injected distributed TrainingBackend for an existing multi-rank process group")
        self.engine = DataParallelPPOActor(OmegaConf.create(ACTOR_CONFIG), self.model, self.optimizer)
        self.updates = 0
        self.initial_snapshot = self.snapshot_id
        self.reference = None

    def prompt_ids(self, prompt):
        ids = super().prompt_ids(prompt)
        if len(ids) > self.max_prompt_tokens: raise ValueError("Student prompt overflow; truncation is forbidden")
        return ids

    def set_reference(self, teacher):
        if self.tokenizer.get_vocab() != teacher.tokenizer.get_vocab():
            raise ValueError("Teacher/student vocabularies differ")
        self.reference = teacher

    def update(self, batch):
        from verl import DataProto
        if batch.metadata.get("behavior_snapshot") != self.snapshot_id:
            raise ValueError("Cannot update from a different behavior-policy snapshot")
        if self.reference is None: raise ValueError("Frozen phase KL reference is required")
        tensors = {k: value.to(self.model.device) for k, value in batch.tensor_data.items()}
        reference_probs = self.torch.zeros_like(tensors["old_log_probs"])
        self.last_update_cost = Cost()
        for i, prompt in enumerate(batch.metadata["student_prompts"]):
            count = int(tensors["response_mask"][i].sum())
            ids = tensors["responses"][i, :count].tolist()
            log_probs, cost = self.reference.score(prompt, ids)
            reference_probs[i, :count] = self.torch.tensor(log_probs, device=self.model.device)
            self.last_update_cost += cost
        tensors["ref_log_prob"] = reference_probs
        data = DataProto.from_dict(tensors=tensors, meta_info={"temperature": 1.0})
        # veRL owns PPO clipping, entropy/KL penalties, accumulation, gradient
        # clipping and optimizer.step. The project provides only advantages.
        metrics = self.engine.update_policy(data)
        self.model.eval()
        self.updates += 1
        self.snapshot_id = f"{self.initial_snapshot}:update:{self.updates}"
        return metrics

    def save_checkpoint(self, output, *, step):
        output.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(output)
        self.tokenizer.save_pretrained(output)
        self.torch.save({"optimizer": self.optimizer.state_dict(), "training_step": step}, output / "optimizer.pt")
        return output
