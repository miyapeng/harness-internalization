from dataclasses import asdict

from ..core.types import write_json


def evaluate_tasks(runner, model, harness, tasks, seeds, output):
    result = runner.rollout(model, harness, tasks, seeds=seeds, output=output)
    expected = {(task, seed) for task in tasks for seed in seeds}
    if {row.key for row in result.evaluations} != expected or len(result.evaluations) != len(expected):
        raise ValueError("Evaluator returned wrong task IDs, seeds or duplicate episodes")
    write_json(output / "episodes.json", [asdict(row) for row in result.evaluations])
    return result
