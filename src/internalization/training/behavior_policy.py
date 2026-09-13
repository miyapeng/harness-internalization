"""Read-only access to the behavior policy until a batch is ready to update.

The serial trainer reuses the actual policy object, rather than copying weights
or loading a cycle-initial teacher. The context must end before optimizer work.
Distributed backends need their own synchronized snapshot implementation.
"""
from __future__ import annotations


class BehaviorPolicySnapshot:
    def __init__(self, policy):
        self.policy = policy
        self.snapshot_id = policy.snapshot_id
        self.active = False

    def __enter__(self):
        if self.active: raise RuntimeError("Behavior snapshot is already active")
        self.model = getattr(self.policy, "model", None)
        self.was_training = self.model.training if self.model is not None else None
        if self.model is not None: self.model.eval()
        self.tensors = self._tensors()
        self.versions = tuple(tensor._version for tensor in self.tensors)
        self.active = True
        self.assert_frozen()
        return self

    def _tensors(self):
        if self.model is None: return ()
        return tuple(self.model.parameters()) + tuple(self.model.buffers())

    def assert_frozen(self):
        if not self.active: raise RuntimeError("Behavior snapshot is outside its batch lifetime")
        if self.policy.snapshot_id != self.snapshot_id:
            raise RuntimeError("Behavior policy changed before module scoring completed")
        if getattr(self.policy, "model", None) is not self.model:
            raise RuntimeError("Behavior model was replaced inside a batch")
        if self.model is not None:
            if self.model.training: raise RuntimeError("Behavior inference must remain in eval mode")
            current = self._tensors()
            if (len(current) != len(self.tensors)
                or any(a is not b for a, b in zip(current, self.tensors))
                or tuple(tensor._version for tensor in current) != self.versions):
                raise RuntimeError("Behavior parameters or buffers changed inside a batch")

    def _call(self, method, *args, **kwargs):
        import torch
        self.assert_frozen()
        with torch.inference_mode():
            result = getattr(self.policy, method)(*args, **kwargs)
        self.assert_frozen()
        return result

    @property
    def pad_token_id(self):
        return self.policy.pad_token_id

    def prompt_ids(self, prompt):
        return self._call("prompt_ids", prompt)

    def generate(self, prompt, *, purpose):
        return self._call("generate", prompt, purpose=purpose)

    def score(self, prompt, response_ids):
        return self._call("score", prompt, response_ids)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if exc_type is None: self.assert_frozen()
        finally:
            self.active = False
            if self.model is not None: self.model.train(self.was_training)
        return False
