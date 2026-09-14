"""Deterministic AppWorld development partitions; scenario families never cross splits."""
import random
import re
from collections import defaultdict
from statistics import mean

from ..core.manifest import TaskManifest


def scenario_groups(ids):
    groups = defaultdict(list)
    if len(ids) != len(set(ids)): raise ValueError("Duplicate AppWorld task IDs")
    for task in ids:
        if not re.fullmatch(r"[a-f0-9]+_[123]", task): raise ValueError("Expected official AppWorld task ID")
        groups[task.rsplit("_", 1)[0]].append(task)
    for scenario, tasks in groups.items():
        if set(tasks) != {f"{scenario}_{i}" for i in (1, 2, 3)}:
            raise ValueError("Keep all three AppWorld scenario variants together")
    return [sorted(groups[key]) for key in sorted(groups)]


def build_manifest(splits, revision, *, seed=42, cycles=3, search_size=15, dev_size=15, cohort_size=30, versioned=True):
    if set(splits) != {"train", "dev", "test_normal", "test_challenge"}:
        raise ValueError("All official AppWorld splits required")
    if cycles < 1 or cohort_size < 30 or any(n <= 0 or n % 3 for n in (search_size, dev_size, cohort_size)):
        raise ValueError("Use complete scenario triples and >=30 tasks per retirement cohort")
    # Validate official IDs/partitions first, including complete test families.
    official = TaskManifest("appworld", revision, {k: tuple(v) for k, v in splits.items()})
    official.validate()
    groups = {name: scenario_groups(ids) for name, ids in splits.items()}
    families = [g[0].rsplit("_", 1)[0] for values in groups.values() for g in values]
    if len(families) != len(set(families)): raise ValueError("Official scenario families overlap between splits")
    rng = random.Random(seed)
    train, dev = groups["train"], groups["dev"]
    rng.shuffle(train); rng.shuffle(dev)
    if len(train) <= search_size//3 or len(dev) < dev_size//3: raise ValueError("Insufficient train/dev tasks")
    search, train = train[:search_size//3], train[search_size//3:]
    chosen_dev, held_out = dev[:dev_size//3], dev[dev_size//3:]
    cohort_names=tuple(f"retirement_{i}" for i in range(cycles)) + (tuple(f"acceptance_{i}" for i in range(cycles)) if versioned else ())
    need = len(cohort_names) * cohort_size//3
    from_train = max(0, need-len(held_out))
    if len(train) <= from_train: raise ValueError("Insufficient non-test data; cannot weaken attribution or reuse cohorts")
    held_out += train[:from_train]
    train = train[from_train:]
    # Any unused official-dev families remain dev, never training or search.
    chosen_dev += held_out[need:]
    held_out = held_out[:need]
    rng.shuffle(held_out)
    flatten = lambda groups: tuple(task for group in groups for task in group)
    partitions = {"train":flatten(train), "search":flatten(search), "dev":flatten(chosen_dev)}
    for index,name in enumerate(cohort_names):
        start = index*cohort_size//3
        partitions[name] = flatten(held_out[start:start+cohort_size//3])
    for name in ("test_normal", "test_challenge"): partitions[name] = tuple(splits[name])
    result = TaskManifest("appworld", revision, partitions)
    result.validate_loop(cycles,versioned=versioned,cohort_minimum=30)
    return result


def aggregate_results(rows):
    if not rows or len({r.key for r in rows}) != len(rows): raise ValueError("Nonempty unique task/seed evaluations required")
    ids = sorted({r.task_id for r in rows})
    seeds = sorted({r.seed for r in rows})
    if {r.key for r in rows} != {(task, seed) for task in ids for seed in seeds}:
        raise ValueError("AppWorld aggregate needs the same seed set per task")
    scores = {r.key:r.success for r in rows}
    result = {"task_goal_completion":100*mean(scores.values()), "scenario_goal_completion":None,
              "tasks":len(ids), "seeds":seeds, "scale":"percent"}
    try: groups = scenario_groups(ids)
    except ValueError:
        result["scenario_metric_status"] = "not_computed_incomplete_scenario_triples"
    else:
        result["scenario_goal_completion"] = 100*mean(min(scores[task, seed] for task in group)
                                                      for group in groups for seed in seeds)
        result["scenario_metric_status"] = "complete_scenario_triples"
    return result
