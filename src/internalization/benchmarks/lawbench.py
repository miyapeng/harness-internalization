"""Native zero-shot LawBench responses, scored by the external official dispatcher."""
from __future__ import annotations

import csv
from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from .common import Catalog, checked_repo, score01
from ..core.types import write_json
from ..training.rollout import EnvironmentStep

CATEGORIES = ("1-1", "1-2", *(f"2-{i}" for i in range(1,11)), *(f"3-{i}" for i in range(1,9)))
STATUS = "implemented_mock_verified_official_evaluator_not_run"


class OfficialLawBenchScorer:
    def __init__(self, config):
        self.root = checked_repo(config.options["evaluator_root"], config.options["evaluator_commit"])
        self.timeout = config.timeout_s

    def score(self, predictions, output):
        """Score category -> list of records. The same call supports complete-category evaluation."""
        output = Path(output).resolve()
        for category, rows in predictions.items():
            if category not in CATEGORIES or not rows: raise ValueError("Unknown/empty LawBench category")
            write_json(output/"predictions"/"student"/(category+".json"), {str(i):r for i,r in enumerate(rows)})
        result_file = output/"results.csv"
        # Upstream 2-1 writes fixed temporary filenames in evaluation/utils.
        # Serialize calls sharing a checkout and use the same worker's python3 for ChERRANT.
        lock_name = hashlib.sha256(str(self.root).encode()).hexdigest()
        env = dict(os.environ)
        env["PATH"] = str(Path(sys.executable).parent)+os.pathsep+env.get("PATH", "")
        with (output/"official.stdout.log").open("x") as log, \
             (Path(tempfile.gettempdir())/("hi-lawbench-"+lock_name+".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            subprocess.run([sys.executable, "main.py", "-i", str(output/"predictions"), "-o", str(result_file)],
                cwd=self.root/"evaluation", stdout=log, stderr=subprocess.STDOUT, check=True, timeout=self.timeout, env=env)
        with result_file.open() as file: rows = list(csv.DictReader(file))
        if len(rows) != len(predictions) or {r["task"] for r in rows} != set(predictions):
            raise RuntimeError("Official LawBench scorer returned incomplete categories")
        return {r["task"]:{"score":score01(r["score"]),
                          "abstention_rate":score01(r["abstention_rate"])} for r in rows}


class LawBenchEnvironment:
    def __init__(self, config, output, *, training=False, scorer=None):
        self.config, self.output, self.training = config, Path(output), training
        self.catalog = Catalog(config.catalog, "lawbench")
        self.scorer = scorer or OfficialLawBenchScorer(config)
        self.done = True

    def reset(self, task_id, seed):
        row = self.catalog.get(task_id, self.training)
        self.record, self.category = row["record"], row["category"]
        if self.category not in CATEGORIES: raise ValueError("Unsupported LawBench category")
        self.task_id, self.seed, self.done = task_id, seed, False
        self.prompt = self.record["instruction"]+"\n"+self.record["question"]
        write_json(self.output/"identity.json", {"task_id":task_id, "seed":seed,
            "catalog_hash":self.catalog.fingerprint, "category":self.category, "setting":"zero_shot"})
        return {"observation":self.prompt}

    def step(self, action):
        if self.done: raise RuntimeError("Episode is already finished")
        prediction = {"origin_prompt":[{"role":"HUMAN", "prompt":self.prompt}],
            "prediction":action, "refr":self.record["answer"]}
        values = self.scorer.score({self.category:[prediction]}, self.output/"official")[self.category]
        self.done = True
        write_json(self.output/"grade.json", {"task_id":self.task_id, "seed":self.seed,
            "category":self.category, "prediction":action, "metrics":values,
            "reward_metric":"official_single_example_score"})
        return asdict(EnvironmentStep(self.prompt+"\nResponse: "+action+"\nResponse submitted.",
            values["score"], True, values["score"], True, 0))

    def close(self): pass
