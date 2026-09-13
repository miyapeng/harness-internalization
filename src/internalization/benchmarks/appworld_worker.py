"""Official AppWorld engine adapter, imported only inside the environment worker.

Project-owned glue, not a copy of upstream environment or benchmark code.
API contract: AppWorld v0.1.3.post1. See docs/benchmarks/APPWORLD.md.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import sys
import traceback

VERSION = "0.1.3.post1"
SPLITS = ("train", "dev", "test_normal", "test_challenge")


class NativeAppWorld:
    def __init__(self, world_class, load_task_ids, root, *, max_steps=30, max_api_calls=1000, timeout=100):
        self.world_class, self.load_task_ids, self.root = world_class, load_task_ids, Path(root)
        self.max_steps, self.max_api_calls, self.timeout = max_steps, max_api_calls, timeout
        self.world = None
        self.done, self.steps = False, 0

    def datasets(self):
        return {"splits": {name: self.load_task_ids(name) for name in SPLITS},
                "package_version": VERSION,
                "split_sha256": {name: hashlib.sha256((self.root / "data/datasets" / (name+".txt")).read_bytes()).hexdigest()
                                 for name in SPLITS}}

    def reset(self, task_id, seed, training, experiment_name):
        if self.world is not None: raise RuntimeError("Worker already has a world")
        allowed = ("train",) if training else SPLITS
        if not any(task_id in self.load_task_ids(name) for name in allowed):
            raise ValueError("Unknown AppWorld task or training outside the official train split")
        if type(seed) is not int or not re.fullmatch(r"hi_[a-f0-9]{32}", experiment_name):
            raise ValueError("Invalid episode seed or unique experiment name")
        # Minimal ground truth is required by world.evaluate(), but it remains
        # in the worker. No ground_truth field or grader output enters prompts.
        self.world = self.world_class(task_id=task_id, experiment_name=experiment_name,
            random_seed=seed, max_interactions=self.max_steps, max_api_calls_per_interaction=self.max_api_calls,
            timeout_seconds=self.timeout, load_ground_truth=True, ground_truth_mode="minimal",
            raise_on_unsafe_syntax=True, null_patch_unsafe_execution=True,
            import_utils=False, parse_datetimes=False, allow_datetime_change=False,
            add_login_shortcut=False, munchify_response=False)
        task = self.world.task
        if task.id != task_id: raise ValueError("Official engine loaded the wrong task")
        public = {"instruction": task.instruction, "supervisor": dict(task.supervisor),
                  "datetime": str(task.datetime), "apps": dict(task.app_descriptions),
                  "helper_api_docs": {name: dict(task.api_docs[name]) for name in ("api_docs", "supervisor")}}
        observation = ("Solve the task by submitting one Python code block per turn to a persistent AppWorld interpreter. "
            "Use the provided apis object and API documentation to discover and call APIs. "
            "Print results you need to inspect. Variables persist between turns. "
            "When finished, call the supervisor complete_task API yourself, providing an answer if required. "
            "Return executable Python, optionally inside a single ```python``` block.\n\n"
            + json.dumps(public, ensure_ascii=False))
        self.world.save_logs()
        self.api_count = self._api_count()
        return {"task_id": task_id, "seed": seed, "observation": observation,
                "provenance": {"task_id":task_id, "seed":seed, "package_version":VERSION,
                    "db_version":task.db_version, "experiment_name":experiment_name,
                    "official_output_directory":str(self.world.output_directory),
                    "max_steps":self.max_steps, "max_api_calls_per_step":self.max_api_calls,
                    "execution_timeout_s":self.timeout, "metric":"official_TestTracker.success",
                    "tool_calls":"execute_invocations_plus_tracked_app_api_requests"}}

    def _api_count(self):
        path = Path(self.world.output_logs_directory) / "api_calls.jsonl"
        return sum(bool(line.strip()) for line in path.read_text().splitlines())

    def step(self, action):
        if self.world is None or self.done: raise RuntimeError("No active AppWorld task")
        if not isinstance(action, str): raise ValueError("Python action must be text")
        self.steps += 1
        code = action.strip()
        block = re.fullmatch(r"```(?:python|py)?\s*\n(.*?)\n```", code, re.DOTALL)
        if block: code = block.group(1)
        # All execution, syntax checks, and safety enforcement belong to the
        # official interpreter, including malformed actions and runtime errors.
        observation = self.world.execute(code)
        if not isinstance(observation, str): raise TypeError("AppWorld execute must return text")
        self.world.save_logs()
        api_count = self._api_count()
        delta = api_count - self.api_count
        if delta < 0: raise RuntimeError("AppWorld API counter moved backwards")
        self.api_count = api_count
        completed = bool(self.world.task_completed())
        self.done = completed or self.steps >= self.max_steps
        success = 0.
        if self.done:
            tracker = self.world.evaluate()
            if tracker.num_tests <= 0 or tracker.total_count != tracker.num_tests:
                raise RuntimeError("AppWorld grader did not run all required tests")
            if type(tracker.success) is not bool: raise TypeError("Expected official boolean task success")
            success = float(tracker.success)
        return {"observation":observation, "reward":success, "success":success, "done":self.done,
                "action_valid":not observation.startswith("Execution failed."),
                "tool_calls":1+delta, "api_calls":delta, "execute_calls":1,
                "termination":"completed" if completed else "step_limit" if self.done else None}

    def close(self):
        if self.world is not None:
            self.world.close()
            self.world = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-api-calls", type=int, default=1000)
    parser.add_argument("--timeout", type=int, default=100)
    args = parser.parse_args()
    wire = sys.stdout
    world = None
    try:
        with redirect_stdout(sys.stderr):
            if version("appworld") != VERSION: raise RuntimeError(f"Install appworld=={VERSION} in the worker environment")
            os.environ["APPWORLD_ROOT"] = str(args.root.resolve(strict=True))
            from appworld import AppWorld, load_task_ids
            world = NativeAppWorld(AppWorld, load_task_ids, args.root,
                max_steps=args.max_steps, max_api_calls=args.max_api_calls, timeout=args.timeout)
        for line in sys.stdin:
            try:
                message = json.loads(line)
                op = message.pop("op")
                if op not in ("reset", "step", "datasets"): raise ValueError("Unsupported environment operation")
                with redirect_stdout(sys.stderr): result = getattr(world, op)(**message)
                wire.write(json.dumps({"ok":True, "result":result}, ensure_ascii=False)+"\n")
                wire.flush()
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                wire.write(json.dumps({"ok":False, "error":f"{type(error).__name__}: {error}"})+"\n")
                wire.flush()
                break
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        wire.write(json.dumps({"ok":False, "error":f"{type(error).__name__}: {error}"})+"\n")
        wire.flush()
    finally:
        if world is not None:
            with redirect_stdout(sys.stderr): world.close()


if __name__ == "__main__": main()
