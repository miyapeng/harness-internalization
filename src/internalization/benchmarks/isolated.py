"""Bounded JSON-lines connection to an external benchmark Python environment."""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import selectors
import subprocess
import time
import uuid

from ..core.types import write_json
from ..training.rollout import EnvironmentStep


class WorkerProcess:
    def __init__(self, config, output):
        self.config, self.output = config, Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        # Preserve venv symlinks: resolving bin/python to the system binary loses the venv.
        python = Path(config.options["python"]).expanduser().absolute()
        if not python.is_file(): raise FileNotFoundError(python)
        config_file = self.output / "config.json"
        write_json(config_file, asdict(config))
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.log = (self.output / "worker.stderr.log").open("x")
        try:
            self.process = subprocess.Popen([str(python), "-u", "-m", "internalization.benchmarks.worker",
                "--config", str(config_file), "--output", str(self.output)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, env=env, shell=False)
        except BaseException:
            self.log.close()
            raise
        self.pending = b""

    def call(self, operation, **payload):
        self.process.stdin.write((json.dumps({"op":operation, **payload})+"\n").encode())
        self.process.stdin.flush()
        deadline = time.monotonic() + self.config.timeout_s
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            while b"\n" not in self.pending:
                left = deadline-time.monotonic()
                if left <= 0 or not selector.select(left): raise TimeoutError(f"Worker timeout; {self.output}")
                data = os.read(self.process.stdout.fileno(), 65536)
                if not data: raise RuntimeError(f"Worker exited; {self.output}")
                self.pending += data
        line, self.pending = self.pending.split(b"\n", 1)
        message = json.loads(line)
        if not message["ok"]: raise RuntimeError(f"Benchmark worker: {message['error']}")
        return message["result"]

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.close()
            try: self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try: self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=10)
        for stream in (self.process.stdin, self.process.stdout):
            if not stream.closed: stream.close()
        self.log.close()


class IsolatedEnvironment:
    def __init__(self, config, output, *, training=False, process_factory=WorkerProcess):
        self.config, self.output, self.training = config, Path(output), training
        self.process_factory, self.process = process_factory, None

    def reset(self, task_id, seed):
        self.process = self.process_factory(self.config, self.output)
        result = self.process.call("reset", task_id=task_id, seed=seed, training=self.training)
        return result["observation"]

    def step(self, action):
        return EnvironmentStep(**self.process.call("step", action=action))

    def close(self):
        if self.process is not None: self.process.close()


def environment_factory(config, output, training=False):
    return lambda: IsolatedEnvironment(config, Path(output)/"environments"/uuid.uuid4().hex, training=training)
