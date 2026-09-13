"""Official benchmark aggregates kept separate from episode-level retirement scores."""
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess

from .common import Catalog
from .lawbench import CATEGORIES
from ..core.types import write_json


def aggregate(config, output, episodes):
    output = Path(output)
    grades = [json.loads(p.read_text()) for p in sorted((output/"environments").glob("*/grade.json"))]
    expected = {e.key for e in episodes}
    if len(grades) != len(expected) or {(g["task_id"],g["seed"]) for g in grades} != expected:
        raise ValueError("Benchmark grades do not match evaluated task/seed pairs")
    if not grades: raise ValueError("No benchmark grades")
    if config.benchmark != "lawbench":
        keys = set(grades[0]["metrics"])
        if any(set(g["metrics"]) != keys for g in grades): raise ValueError("Metric schema mismatch")
        return {"metrics":{k:sum(g["metrics"][k] for g in grades)/len(grades) for k in sorted(keys)},
                "episodes":len(grades), "tasks":len({g["task_id"] for g in grades})}
    return aggregate_lawbench(config, output, grades)


def aggregate_lawbench(config, output, grades):
    """Also accepts unscored native predictions, including officially skipped singletons."""
    output = Path(output)
    if not grades or len({(g["task_id"],g["seed"]) for g in grades}) != len(grades):
        raise ValueError("Nonempty unique LawBench prediction identities required")
    # Official functions may pool examples nonlinearly: never substitute mean singleton scores.
    catalog = Catalog(config.catalog, "lawbench")
    groups = defaultdict(lambda:defaultdict(list))
    for grade in grades:
        record = catalog.get(grade["task_id"])["record"]
        groups[grade["seed"]][grade["category"]].append({"origin_prompt":[{"role":"HUMAN",
            "prompt":record["instruction"]+"\n"+record["question"]}], "prediction":grade["prediction"], "refr":record["answer"]})
    results = {}
    for seed, predictions in groups.items():
        folder = output/"lawbench_aggregate"/str(seed)
        request, response = folder/"request.json", folder/"response.json"
        write_json(request, {"config":config.__dict__, "predictions":predictions})
        env = dict(os.environ)
        env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])+os.pathsep+env.get("PYTHONPATH", "")
        with (folder/"worker.log").open("x") as log:
            subprocess.run([config.options["python"], "-m", "internalization.benchmarks.lawbench_scoring",
                "--request", str(request.resolve()), "--response", str(response.resolve())],
                check=True, stdout=log, stderr=subprocess.STDOUT, timeout=config.timeout_s, env=env)
        results[str(seed)] = json.loads(response.read_text())
    categories = set.intersection(*(set(r) for r in results.values()))
    per_category = {c:{k:sum(r[c][k] for r in results.values())/len(results)
                       for k in ("score", "abstention_rate")} for c in sorted(categories)}
    macro = sum(r["score"] for r in per_category.values())/len(per_category)
    return {"per_category":per_category, "per_seed":results, "macro_observed_categories":macro,
        "official_20_category_macro":macro if categories == set(CATEGORIES) else None,
        "full_dataset_verified":all(sum(g["category"]==c and g["seed"]==seed for g in grades)==500
                                    for c in CATEGORIES for seed in groups), "episodes":len(grades)}
