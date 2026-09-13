"""Native LawBench final evaluation: generate all predictions, then official category scoring.

No artificial per-example reward is assigned to officially skipped/undefined items.
This evaluator is deliberately outside the training and retirement TaskRunner.
"""
from dataclasses import asdict
from pathlib import Path
import time

from .aggregate import aggregate_lawbench
from .common import Catalog
from ..core.types import Cost, State, write_json
from ..harness.runtime import TeacherHarness


def evaluate(config, model, harness, tasks, seeds, output):
    if config.benchmark != "lawbench": raise ValueError("Expected LawBench configuration")
    if not tasks or not seeds or len(set(tasks))!=len(tasks) or len(set(seeds))!=len(seeds):
        raise ValueError("Unique nonempty tasks/seeds required")
    catalog = Catalog(config.catalog,"lawbench",config.options.get("catalog_hash"))
    catalog.check_selection(tasks,final=True)
    grades, cost, started = [], Cost(), time.monotonic()
    for index, task in enumerate(tasks):
        row = catalog.get(task)
        public = row["record"]["instruction"]+"\n"+row["record"]["question"]
        for seed in seeds:
            state = State(task,f"{task}:{seed}",0,public)
            runtime = TeacherHarness(model,harness)
            advice = runtime.advise(state)
            response = model.generate(advice.teacher_prompt,purpose="action")
            episode_cost = advice.cost+response.cost
            cost += episode_cost
            grade = {"task_id":task,"seed":seed,"category":row["category"],"prediction":response.text}
            grades.append(grade)
            write_json(Path(output)/"predictions"/f"{index:06d}_{seed}.json",{**grade,
                "student_prompt":advice.teacher_prompt,"response_ids":response.response_ids,
                "model_snapshot":model.snapshot_id,"harness_version":harness.version,
                "cost":asdict(episode_cost),"per_example_reward":None})
    values = aggregate_lawbench(config,output,grades)
    fields = asdict(cost)
    fields["latency_s"] = time.monotonic()-started
    return {"benchmark":"lawbench","benchmark_metrics":values,"cost":fields,
        "catalog_hash":catalog.fingerprint,"per_example_reward_assigned":False,"selection":False}
