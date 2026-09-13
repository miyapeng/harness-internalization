"""Import local official data and pin identities; never invent a published training split."""
import csv
import json
from pathlib import Path

from .common import sha256, tree_hash
from .lawbench import CATEGORIES
from ..core.manifest import TaskManifest
from ..core.types import digest


def import_tasks(benchmark, source, *, split="test", image_lock=None):
    source = Path(source).resolve(strict=True)
    rows = []
    if benchmark == "terminalbench2":
        for path in sorted(source.iterdir()):
            if not path.is_dir() or not (path/"task.toml").is_file(): continue
            if not (path/"instruction.md").is_file() or not (path/"tests"/"test.sh").is_file():
                raise ValueError("Incomplete Harbor task bundle")
            rows.append({"id":path.name, "split":split, "task_path":str(path), "task_hash":tree_hash(path)})
    elif benchmark == "hotpotqa":
        fingerprint = sha256(source)
        for record in json.loads(source.read_text()):
            if not all(k in record for k in ("_id", "question", "context", "answer", "supporting_facts")):
                raise ValueError("Labelled HotpotQA distractor records required")
            rows.append({"id":record["_id"], "split":split, "record":record, "source_hash":fingerprint})
    elif benchmark == "lawbench":
        for category in CATEGORIES:
            path = source/(category+".json")
            if not path.exists(): continue
            fingerprint = sha256(path)
            for index, record in enumerate(json.loads(path.read_text())):
                if not all(isinstance(record.get(k), str) for k in ("instruction", "question", "answer")):
                    raise ValueError("Malformed LawBench record")
                rows.append({"id":f"{category}:{fingerprint[:16]}:{index}", "category":category,
                    "row_index":index, "source_hash":fingerprint, "split":split, "record":record})
    elif benchmark == "swebench_pro":
        fingerprint = sha256(source)
        if source.suffix == ".jsonl": records = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        elif source.suffix == ".json": records = json.loads(source.read_text())
        else:
            with source.open() as file: records = list(csv.DictReader(file))
        if image_lock is None: raise ValueError("A Pro instance image lock is required")
        for record in records:
            for key in ("instance_id", "repo", "base_commit", "problem_statement", "fail_to_pass", "pass_to_pass",
                        "before_repo_set_cmd", "selected_test_files_to_run"):
                if key not in record: raise ValueError(f"Missing Pro field: {key}")
            # The official evaluator parses these serialized list columns from its dataframe.
            record = dict(record)
            for key in ("fail_to_pass", "pass_to_pass", "selected_test_files_to_run"):
                if isinstance(record[key], list): record[key] = repr(record[key])
            lock = image_lock[record["instance_id"]]
            rows.append({"id":record["instance_id"], "split":split, "record":record,
                "image":lock["digest"], "image_tag":lock["tag"], "source_hash":fingerprint})
    else: raise ValueError("Unknown benchmark")
    if not rows: raise ValueError("No benchmark tasks found")
    return rows


def make_catalog_manifest(benchmark, revision, test, train=(), *, cohort_size=30):
    if not revision or revision in ("main", "master", "latest"):
        raise ValueError("A pinned dataset revision is required")
    if any(r["split"] != "test" for r in test) or any(r["split"] != "train" for r in train):
        raise ValueError("Source splits do not match requested train/test roles")
    rows = list(train)+list(test)
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids): raise ValueError("Task IDs overlap across sources")
    # Detect identical problems even if a second file assigns them different IDs.
    identities = [r.get("task_hash") or digest({k:r["record"].get(k) for k in
        ("question", "instruction", "problem_statement", "repo", "base_commit")}) for r in rows]
    if set(identities[:len(train)]) & set(identities[len(train):]):
        raise ValueError("Duplicate problems across task sources")
    partitions = {"test":tuple(r["id"] for r in test)}
    if train:
        if cohort_size < 30: raise ValueError("Do not lower the 30-task attribution/retirement threshold")
        pool = sorted(r["id"] for r in train)
        if len(pool) < 6*cohort_size: raise ValueError("Separate training source needs at least six task cohorts")
        for name in ("search", "dev", "retirement_0", "retirement_1", "retirement_2"):
            partitions[name], pool = tuple(pool[:cohort_size]), pool[cohort_size:]
        partitions["train"] = tuple(pool)
    catalog = {"schema_version":1, "benchmark":benchmark, "revision":revision, "tasks":rows}
    manifest = TaskManifest(benchmark, revision, partitions)
    manifest.validate()
    return catalog, manifest
