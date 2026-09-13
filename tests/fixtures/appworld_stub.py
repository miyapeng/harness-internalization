"""Explicit test double for AppWorld's public API; no official tasks or engine code."""
import json
import os
from pathlib import Path
from types import SimpleNamespace


def load_task_ids(split):
    families = {"train":"aaaaaaa", "dev":"bbbbbbb", "test_normal":"ccccccc", "test_challenge":"ddddddd"}
    return [f"{families[split]}_{i}" for i in (1, 2, 3)]


class PublicTask:
    def __init__(self, task_id):
        self.id, self.instruction, self.db_version = task_id, "Complete the synthetic task.", "stub-db-v1"
        self.supervisor = {"first_name":"Test", "last_name":"User", "email":"test@example.test", "phone_number":"000"}
        self.datetime = "2024-01-01"
        self.app_descriptions = {"api_docs":"API documentation", "supervisor":"Task owner"}
        self.api_docs = {"api_docs":{"show_api_doc":{"description":"Discover APIs"}},
                         "supervisor":{"complete_task":{"description":"Mark completion"}}}

    @property
    def ground_truth(self): raise AssertionError("Hidden ground truth must not be read by the adapter")


class AppWorld:
    created = []
    def __init__(self, **kwargs):
        type(self).created.append(self)
        self.kwargs, self.task = kwargs, PublicTask(kwargs["task_id"])
        self.output_directory = str(Path(os.environ["APPWORLD_ROOT"]) / "experiments/outputs" / kwargs["experiment_name"] / "tasks" / self.task.id)
        self.output_logs_directory = str(Path(self.output_directory) / "logs")
        Path(self.output_logs_directory).mkdir(parents=True, exist_ok=False)
        self.calls, self.executed, self.evaluations, self.closed = [], [], 0, False
        self.complete, self.pass_tests = False, True
        self.save_logs()

    def execute(self, code):
        self.executed.append(code)
        if code == 'print("hello")':
            self.calls.extend([{"api":"lookup"}, {"api":"read"}])
            return "hello"
        if code == "apis.supervisor.complete_task()":
            self.calls.append({"api":"complete_task"})
            self.complete = True
            return "Task marked complete."
        return "Execution failed. Traceback:\nNameError: invalid synthetic action"

    def save_logs(self):
        (Path(self.output_logs_directory) / "api_calls.jsonl").write_text("".join(json.dumps(row)+"\n" for row in self.calls))

    def task_completed(self): return self.complete

    def evaluate(self):
        self.evaluations += 1
        # Deliberately include forbidden details; the adapter must read only
        # public metric properties, never forward grader diagnostics.
        return SimpleNamespace(success=self.pass_tests and self.complete, num_tests=2, total_count=2,
                               failures=["SECRET_GRADER_ANSWER"], difficulty="SECRET_DIFFICULTY")

    def close(self): self.closed = True
