"""AppWorld environment bridge. The official engine runs in its own Python process."""
from __future__ import annotations

import json
import os
import selectors
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from ..core.types import Journal, write_json
from ..training.rollout import EnvironmentStep

PACKAGE_VERSION = "0.1.3.post1"
STATUS = "implemented_mock_verified_real_environment_not_run"


@dataclass(frozen=True)
class AppWorldConfig:
    python_env_var: str = "HI_APPWORLD_PYTHON"
    root_env_var: str = "APPWORLD_ROOT"
    max_steps: int = 30
    max_api_calls_per_step: int = 1000
    execution_timeout_s: int = 100
    rpc_timeout_s: int = 180
    max_context: int = 32768
    max_prompt_tokens: int = 28672
    max_action_tokens: int = 1024
    max_new_tokens: int = 256

    def __post_init__(self):
        for value in (self.max_steps, self.max_api_calls_per_step, self.execution_timeout_s, self.rpc_timeout_s,
                      self.max_context, self.max_prompt_tokens, self.max_action_tokens, self.max_new_tokens):
            if type(value) is not int or value <= 0: raise ValueError("Positive AppWorld limits required")
        if self.rpc_timeout_s <= self.execution_timeout_s:
            raise ValueError("RPC timeout must exceed execution timeout")
        if self.max_prompt_tokens + max(self.max_action_tokens, self.max_new_tokens) > self.max_context:
            raise ValueError("Prompt and generation budgets exceed AppWorld context limit")

    @property
    def model_options(self):
        return {name: getattr(self, name) for name in ("max_context", "max_action_tokens", "max_new_tokens")}

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class AppWorldProcess:
    """JSON-lines RPC; no shell invocation and no AppWorld imports in the model process."""
    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        executable = os.environ.get(config.python_env_var)
        root = os.environ.get(config.root_env_var)
        if not executable or not root:
            raise ValueError(f"Set {config.python_env_var} and {config.root_env_var} for the isolated AppWorld environment")
        executable = str(Path(executable).resolve(strict=True))
        root = str(Path(root).resolve(strict=True))
        env = dict(os.environ)
        source = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["APPWORLD_ROOT"] = root
        self.log = (self.output / "worker.stderr.log").open("x")
        try:
            self.process = subprocess.Popen([executable, "-u", "-m", "internalization.benchmarks.appworld_worker",
                "--root", root, "--max-steps", str(config.max_steps),
                "--max-api-calls", str(config.max_api_calls_per_step),
                "--timeout", str(config.execution_timeout_s)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env, shell=False)
        except BaseException:
            self.log.close()
            raise
        self.pending = b""

    def call(self, operation, **payload):
        self.process.stdin.write((json.dumps({"op": operation, **payload}) + "\n").encode())
        self.process.stdin.flush()
        deadline = time.monotonic() + self.config.rpc_timeout_s
        # Read bytes directly: a TextIOWrapper can hide already-buffered lines
        # from select(), especially on a fast local subprocess.
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self.pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError(f"AppWorld {operation} timed out; see {self.output}")
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data: raise RuntimeError(f"AppWorld worker exited; see {self.output}")
                self.pending += data
        line, self.pending = self.pending.split(b"\n", 1)
        result = json.loads(line)
        if not result.get("ok"):
            raise RuntimeError(f"AppWorld worker {operation} failed: {result.get('error')}")
        return result["result"]

    def close(self):
        if self.process.poll() is None:
            # EOF allows finally: world.close(); bounded shutdown also handles
            # a wedged interpreter. No output/checkpoint files are removed.
            self.process.stdin.close()
            try: self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try: self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        for stream in (self.process.stdin, self.process.stdout):
            if not stream.closed: stream.close()
        self.log.close()


class AppWorldEnvironment:
    def __init__(self, config, output, *, training=False, process_factory=AppWorldProcess):
        self.config, self.output, self.training = config, Path(output), training
        self.process_factory, self.process = process_factory, None
        self.finished = False

    def reset(self, task_id, seed):
        if self.process is not None: raise RuntimeError("Use a fresh environment for each task/seed")
        self.process = self.process_factory(self.config, self.output)
        info = self.process.call("reset", task_id=task_id, seed=seed, training=self.training,
                                 experiment_name="hi_" + uuid.uuid4().hex)
        if info["task_id"] != task_id or info["seed"] != seed:
            raise ValueError("AppWorld returned a different task or seed")
        write_json(self.output / "identity.json", info["provenance"])
        self.journal = Journal(self.output / "environment.jsonl")
        self.history = info["observation"]
        return self.history

    def step(self, action):
        if self.process is None or self.finished: raise RuntimeError("No active AppWorld episode")
        result = self.process.call("step", action=action)
        self.finished = result["done"]
        # Only code/REPL output go into shared history. Grader reports and
        # numeric success/reward are never appended to model input.
        self.history += "\n\n[Submitted Python]\n" + action + "\n[Execution output]\n" + result["observation"]
        self.journal.append("step", **result)
        return EnvironmentStep(self.history, result["reward"], self.finished, result["success"],
                               result["action_valid"], result["tool_calls"])

    def close(self):
        if self.process is not None:
            self.process.close()
            self.process = None


def environment_factory(config, output, *, training=False):
    return lambda: AppWorldEnvironment(config, Path(output) / "appworld" / uuid.uuid4().hex, training=training)
