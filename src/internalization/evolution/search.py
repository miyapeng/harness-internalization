from dataclasses import asdict

from ..core.types import write_json
from ..harness.module import Harness
from ..evaluation.retirement import paired_interval


def evaluate_tasks(runner, model, harness, tasks, seeds, output):
    result = runner.rollout(model, harness, tasks, seeds=seeds, output=output)
    expected = {(task, seed) for task in tasks for seed in seeds}
    if {row.key for row in result.evaluations} != expected or len(result.evaluations) != len(expected):
        raise ValueError("Evaluator returned wrong task IDs, seeds or duplicate episodes")
    write_json(output / "episodes.json", [asdict(row) for row in result.evaluations])
    return result


def search_candidates(proposer, runner, request, *, dev_tasks, baseline_search, baseline_dev,
                      seeds, policy, archive, journal):
    """Return the winning Candidate with lineage, after module validation."""
    candidates = proposer.propose(request)
    if len(candidates) != request.count: raise ValueError("Proposer returned wrong candidate count")
    accepted = []
    for index, candidate in enumerate(candidates):
        if candidate.parent != request.harness.version or candidate.version != request.cycle or candidate.index != index:
            raise ValueError("Candidate lineage does not match the proposal request")
        try:
            module = candidate.module()
            full = Harness(request.harness.modules + (module,))
        except (ValueError, SyntaxError, TypeError) as exc:
            archive.record(candidate, status="invalid", reason=str(exc))
            journal.append("candidate_rejected", cycle=request.cycle, candidate=index, reason=str(exc))
            continue
        folder = request.output.parent / f"candidate_{index}"
        search = evaluate_tasks(runner, request.checkpoint, full, request.tasks, seeds, folder / "search")
        dev = evaluate_tasks(runner, request.checkpoint, full, dev_tasks, seeds, folder / "dev")
        sg = paired_interval(list(search.evaluations), list(baseline_search.evaluations), lambda r: r.success, policy)
        dg = paired_interval(list(dev.evaluations), list(baseline_dev.evaluations), lambda r: r.success, policy)
        journal.append("candidate_score", cycle=request.cycle, candidate=index, module_version=module.version,
                       search_gain=sg, dev_gain=dg)
        useful = sg["low"] > 0 and dg["low"] > 0
        archive.record(candidate, status="eligible" if useful else "rejected", search_gain=sg, dev_gain=dg)
        if useful:
            accepted.append((dg["mean"], -sum(r.cost.total_tokens for r in dev.evaluations), candidate))
    return max(accepted, key=lambda item: item[:2])[2] if accepted else None
