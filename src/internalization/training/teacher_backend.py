from __future__ import annotations

import time
from pathlib import Path

from ..core.types import Cost, digest
from ..harness.runtime import Completion


def checkpoint_fingerprint(path: Path):
    """Hash model file metadata to detect changed local snapshots cheaply.

    Run provenance should additionally record artifact storage checksums. This
    fingerprint is not a cryptographic hash of multi-GB weight contents.
    """
    path = path.resolve(strict=True)
    files = sorted(p for p in path.rglob("*") if p.is_file() and
                   (p.suffix in (".safetensors", ".bin", ".json", ".model")))
    if not any(p.suffix in (".safetensors", ".bin") for p in files):
        raise ValueError("Teacher checkpoint must be a local HF model with weight files")
    return digest([(str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns) for p in files])


class FrozenHFBackend:
    """Separate, eval-only HF model; never aliases the updating student.

    Minimal correctness backend, not a high-throughput production scorer.
    Dependencies load only when a real teacher is requested.
    """
    def __init__(self, checkpoint: str, *, device="cpu", max_new_tokens=192, max_action_tokens=512,
                 max_context=8192, chat_template_kwargs=None, student_tokenizer=None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.path = Path(checkpoint).resolve(strict=True)
        self.snapshot_id = checkpoint_fingerprint(self.path)
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path), local_files_only=True)
        if student_tokenizer is not None and self.tokenizer.get_vocab() != student_tokenizer.get_vocab():
            raise ValueError("Teacher and student must use identical token ID vocabularies")
        self.model = AutoModelForCausalLM.from_pretrained(
            str(self.path), local_files_only=True, torch_dtype="auto").to(device).eval()
        self.model.requires_grad_(False)
        self.max_new_tokens, self.max_context = max_new_tokens, max_context
        self.max_action_tokens = max_action_tokens
        self.chat_template_kwargs = chat_template_kwargs or {}

    def assert_frozen(self):
        if checkpoint_fingerprint(self.path) != self.snapshot_id:
            raise RuntimeError("Teacher checkpoint files changed during the phase")
        if self.model.training or any(p.requires_grad for p in self.model.parameters()):
            raise RuntimeError("Teacher is no longer frozen")

    def prompt_ids(self, prompt):
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False,
            add_generation_prompt=True, **self.chat_template_kwargs)
        return self.tokenizer.encode(rendered, add_special_tokens=False)

    def _sync(self):
        if self.model.device.type == "cuda": self.torch.cuda.synchronize(self.model.device)

    def generate(self, prompt: str, *, purpose: str):
        ids = self.prompt_ids(prompt)
        output_limit = self.max_action_tokens if purpose in ("action", "rollout_action") else self.max_new_tokens
        if len(ids) + output_limit > self.max_context:
            raise ValueError("Teacher context overflow; truncation would break information parity")
        tokens = self.torch.tensor([ids], device=self.model.device)
        self._sync()
        start = time.perf_counter()
        with self.torch.inference_mode():
            sampling = {"temperature": 1.0, "top_p": 1.0, "top_k": 0} if purpose == "rollout_action" else {}
            outputs = self.model.generate(input_ids=tokens, attention_mask=self.torch.ones_like(tokens),
                      max_new_tokens=output_limit, do_sample=(purpose == "rollout_action"),
                      pad_token_id=self.tokenizer.eos_token_id, **sampling)
        self._sync()
        generated = outputs[0, len(ids):].tolist()
        return Completion(self.tokenizer.decode(generated, skip_special_tokens=True),
                          Cost(len(ids), len(generated), 1, int(purpose not in ("action", "rollout_action")),
                               latency_s=time.perf_counter() - start), tuple(generated))

    def score(self, prompt: str, response_ids: list[int]):
        ids = self.prompt_ids(prompt)
        if not ids or len(ids) + len(response_ids) > self.max_context:
            raise ValueError("Teacher scoring context overflow")
        if not response_ids: return [], Cost()
        tokens = self.torch.tensor([ids + response_ids], device=self.model.device)
        self._sync()
        start = time.perf_counter()
        with self.torch.inference_mode():
            logits = self.model(input_ids=tokens, attention_mask=self.torch.ones_like(tokens)).logits
            logits = logits[0, len(ids)-1:-1, :].float()
            targets = self.torch.tensor(response_ids, device=logits.device)
            lp = logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1).cpu().tolist()
        self._sync()
        # Scoring consumes observed response tokens; it generates no response.
        return lp, Cost(len(ids) + len(response_ids), 0, 1, 1,
                        latency_s=time.perf_counter() - start)
