"""SWE-bench Pro public instances: container interaction and official patch evaluation."""
from __future__ import annotations

from dataclasses import asdict
import importlib
import json
from pathlib import Path
import re
import subprocess
import sys

from .common import Catalog, action_json, checked_repo, score01
from ..core.types import write_json
from ..training.rollout import EnvironmentStep

STATUS = "implemented_mock_verified_docker_and_official_tests_not_run"


class DockerWorkspace:
    """Only student commands enter the fresh container; no host/task/grader mounts."""
    def __init__(self, image, timeout, *, client=None):
        if not re.search(r"@sha256:[0-9a-f]{64}$", image):
            raise ValueError("SWE-bench Pro agent image must be pinned by digest")
        if client is None:
            import docker
            client = docker.from_env()
        self.client, self.timeout = client, timeout
        # Require an explicitly prepared image; no silent mutable-tag pull.
        client.images.get(image)
        self.container = client.containers.run(image, command=["-c", "sleep infinity"],
            entrypoint="/bin/bash", detach=True, network_mode="none", working_dir="/app")

    def execute(self, command):
        # timeout is inside the container so timed-out shell processes are also stopped.
        result = self.container.exec_run(["timeout", "--kill-after=5", str(self.timeout),
            "/bin/bash", "-lc", command], workdir="/app", demux=True)
        out, err = result.output
        return {"stdout":(out or b"").decode(errors="replace"),
            "stderr":(err or b"").decode(errors="replace"), "return_code":result.exit_code}

    def close(self):
        # Stop our own ephemeral sandbox. Preserve it for inspection; never prune containers/data.
        self.container.stop(timeout=5)
        self.client.close()


class OfficialProScorer:
    def __init__(self, config):
        self.config = config
        self.root = checked_repo(config.options["evaluator_root"], config.options["evaluator_commit"])

    def image_uri(self, record):
        # The installed evaluator owns its image naming convention.
        sys.path.insert(0, str(self.root))
        try:
            helper = importlib.import_module("helper_code.image_uri")
            return helper.get_dockerhub_image_uri(record["instance_id"],
                self.config.options["dockerhub_username"], record["repo"])
        finally: sys.path.pop(0)

    def assert_image(self, record, image):
        import docker
        client = docker.from_env()
        try:
            current = client.images.get(self.image_uri(record))
            pinned = client.images.get(image)
            if current.id != pinned.id: raise ValueError("Official grader image tag changed from the pinned agent image")
        finally: client.close()

    def score(self, record, patch, output, *, image):
        self.assert_image(record, image)
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        samples, patches = output/"sample.jsonl", output/"patches.json"
        samples.write_text(json.dumps(record)+"\n")
        write_json(patches, [{"instance_id":record["instance_id"], "patch":patch, "prefix":"student"}])
        with (output/"official.stdout.log").open("x") as log:
            subprocess.run([sys.executable, "swe_bench_pro_eval.py", "--raw_sample_path", str(samples),
                "--patch_path", str(patches), "--output_dir", str(output/"results"),
                "--scripts_dir", str(self.root/"run_scripts"), "--num_workers", "1",
                "--dockerhub_username", self.config.options["dockerhub_username"],
                "--use_local_docker", "--block_network"], cwd=self.root, stdout=log,
                stderr=subprocess.STDOUT, check=True, timeout=self.config.timeout_s)
        # Upstream may pull its tag; reject the run if that changes image identity.
        self.assert_image(record, image)
        results = json.loads((output/"results"/"eval_results.json").read_text())
        # The upstream CLI maps infrastructure exceptions to False: do not silently accept those.
        tests_file = output/"results"/record["instance_id"]/"student_output.json"
        tests = json.loads(tests_file.read_text())["tests"]
        if not tests or not all("name" in r and "status" in r for r in tests):
            raise RuntimeError("Missing/invalid official test output")
        if type(results.get(record["instance_id"])) is not bool:
            raise RuntimeError("Missing official resolved result")
        return score01(results[record["instance_id"]])


class SWEBenchProEnvironment:
    def __init__(self, config, output, *, training=False, workspace_factory=DockerWorkspace, scorer=None):
        self.config, self.output, self.training = config, Path(output), training
        self.catalog = Catalog(config.catalog, "swebench_pro")
        self.workspace_factory, self.scorer = workspace_factory, scorer or OfficialProScorer(config)
        self.workspace, self.done = None, True

    def reset(self, task_id, seed):
        row = self.catalog.get(task_id, self.training)
        self.record = row["record"]
        if not re.fullmatch(r"[a-zA-Z0-9_.-]+", task_id): raise ValueError("Invalid instance ID")
        base = self.record["base_commit"]
        if not re.fullmatch(r"[0-9a-f]{40}", base): raise ValueError("Pinned base commit required")
        image = self.image = row["image"]
        expected = self.scorer.image_uri(self.record)
        if row["image_tag"] != expected: raise ValueError("Image lock does not match official Pro instance mapping")
        self.workspace = self.workspace_factory(image, self.config.command_timeout_s)
        setup = self.workspace.execute("git checkout --detach "+base)
        if setup["return_code"] != 0: raise RuntimeError("Cannot initialize Pro base commit")
        clean = self.workspace.execute("git status --porcelain")
        if clean["return_code"] or clean["stdout"].strip(): raise RuntimeError("Pro base workspace is not clean")
        self.task_id, self.seed, self.steps, self.done = task_id, seed, 0, False
        public = self.record["problem_statement"]
        for key in ("requirements", "interface"):
            if self.record.get(key): public += "\n"+str(self.record[key])
        self.history = [public, 'Use {"action":"exec","command":"..."} to work in /app. '
            'Use {"action":"final"} to submit the working tree patch. Each command starts a new shell; files persist.']
        write_json(self.output/"identity.json", {"task_id":task_id, "seed":seed,
            "catalog_hash":self.catalog.fingerprint, "base_commit":base, "image":image,
            "evaluator_commit":self.config.options.get("evaluator_commit"), "network":"none"})
        return {"observation":"\n".join(self.history)}

    def step(self, action):
        if self.done: raise RuntimeError("Episode is already finished")
        self.steps += 1
        valid, calls = True, 0
        try:
            item = action_json(action)
            if item["action"] == "exec":
                if not isinstance(item["command"], str): raise ValueError("Command must be a string")
                result = self.workspace.execute(item["command"])
                calls, obs = 1, json.dumps(result)
            elif item["action"] == "final": self.done, obs = True, "Patch submitted."
            else: raise ValueError("Unknown action")
        except (ValueError, KeyError, TypeError) as exc:
            valid, obs = False, "Invalid action: "+str(exc)
        self.done = self.done or self.steps >= self.config.max_steps
        reward = 0.
        if self.done:
            diff = self.workspace.execute("git add -A && git diff --cached --binary "+self.record["base_commit"])
            if diff["return_code"]: raise RuntimeError("Cannot extract submitted patch")
            calls += 1
            reward = self.scorer.score(self.record, diff["stdout"], self.output/"official", image=self.image)
            write_json(self.output/"grade.json", {"task_id":self.task_id, "seed":self.seed,
                "prediction":diff["stdout"], "metrics":{"resolved":reward}, "reward_metric":"resolved"})
        self.history.extend(["Action: "+action, "Observation: "+obs])
        return asdict(EnvironmentStep("\n".join(self.history), reward, self.done, reward, valid, calls))

    def close(self):
        if self.workspace is not None: self.workspace.close()
