"""Strict, resolved execution settings carried across the process boundary.

Defaults live here. Benchmark limits are inherited once, before explicit overrides;
workers consume the resolved document, never override it with local defaults/env.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .types import write_json

DEFAULTS = {
    "mode": "internalization",
    "max_steps": 30,
    "supervision": "targeted",
    "tasks_per_batch": 4,
    "rollouts_per_task": 2,
    "device": "cuda:0",
    "reference_device": "cpu",
    "model": {"max_context": 8192, "max_prompt_tokens": 4096,
              "max_action_tokens": 512, "max_new_tokens": 192},
    "advantage": {"module_weight": .001, "mode": "mean_std_norm", "normalize_module": False,
                  "clip_module": None, "invalid_action_penalty": .1, "epsilon": 1e-6},
    "optimizer": {"learning_rate": 1e-6, "weight_decay": .01},
}


def _merge(base, changes, path="execution"):
    if not isinstance(changes, dict): raise ValueError(f"{path} must be an object")
    for key, value in changes.items():
        if key not in base: raise ValueError(f"Unknown configuration field: {path}.{key}")
        if isinstance(base[key], dict): _merge(base[key], value, path+"."+key)
        else: base[key] = value


def resolve_execution(raw=None, *, benchmark_limits=None):
    value = deepcopy(DEFAULTS)
    if benchmark_limits: _merge(value, benchmark_limits)
    _merge(value, {} if raw is None else raw)
    if value["mode"] not in ("internalization", "evolution_only"): raise ValueError("Unknown execution mode")
    if value["supervision"] not in ("targeted", "all"): raise ValueError("Unknown supervision mode")
    for key in ("max_steps", "tasks_per_batch", "rollouts_per_task"):
        if type(value[key]) is not int or value[key] < 1: raise ValueError(f"Positive integer required: {key}")
    model = value["model"]
    for key, n in model.items():
        if type(n) is not int or n < 1: raise ValueError(f"Positive integer required: model.{key}")
    if model["max_prompt_tokens"] + max(model["max_action_tokens"], model["max_new_tokens"]) > model["max_context"]:
        raise ValueError("Model context budget exceeded")
    for key in ("device", "reference_device"):
        if not isinstance(value[key], str) or not value[key]: raise ValueError(f"Invalid {key}")
    adv = value["advantage"]
    if adv["mode"] not in ("mean_norm", "mean_std_norm"): raise ValueError("Unknown outcome normalization")
    if type(adv["normalize_module"]) is not bool: raise ValueError("normalize_module must be boolean")
    for key, n in {**{k:adv[k] for k in ("module_weight", "invalid_action_penalty", "epsilon")}, **value["optimizer"]}.items():
        if type(n) not in (int,float) or not math.isfinite(n) or n < 0:
            raise ValueError(f"Invalid numeric configuration: {key}")
    if adv["epsilon"] <= 0 or value["optimizer"]["learning_rate"] <= 0: raise ValueError("epsilon/lr must be positive")
    if adv["clip_module"] is not None and (type(adv["clip_module"]) not in (int,float) or
            not math.isfinite(adv["clip_module"]) or adv["clip_module"] <= 0): raise ValueError("Invalid module clip")
    return value


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def save_effective(output, config):
    path = Path(output)/"effective_config.json"
    if path.exists():
        if json.loads(path.read_text()) != config: raise ValueError("Recorded effective configuration changed")
    else: write_json(path, config)
    checksum = Path(output)/"effective_config.sha256"
    expected = config_hash(config)+"\n"
    if checksum.exists():
        if checksum.read_text() != expected: raise ValueError("Recorded effective hash changed")
    else:
        with checksum.open("x") as stream: stream.write(expected)


def request_execution(request):
    raw = request.get("effective_config")
    if raw is None: return None  # Historical standalone requests only; CommandBackend always sends it.
    resolved = resolve_execution(raw)
    if raw != resolved or request.get("effective_config_hash") != config_hash(resolved):
        raise ValueError("Incomplete or mismatched effective configuration")
    return resolved


class NoActorUpdates(RuntimeError):
    """A completed collection phase with zero actor updates is not a trained model."""
    def __init__(self, summary):
        super().__init__("no_actor_updates")
        self.summary = summary
