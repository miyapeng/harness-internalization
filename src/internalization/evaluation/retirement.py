"""Four-cell, paired evaluation. Bootstrap clusters are task IDs, not steps."""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from statistics import mean

from ..core.types import EpisodeResult


@dataclass(frozen=True)
class RetirementPolicy:
    performance_margin: float = 0.02
    confidence: float = 0.95
    bootstrap_samples: int = 2000
    min_tasks: int = 30
    min_token_saving_fraction: float = 0.05
    seed: int = 42

    def __post_init__(self):
        if not (0 <= self.performance_margin < 1 and 0 < self.confidence < 1
                and 0 <= self.min_token_saving_fraction < 1
                and self.min_tasks >= 2 and self.bootstrap_samples >= 100):
            raise ValueError("Invalid retirement policy")


def _indexed(rows):
    indexed = {r.key: r for r in rows}
    if not rows or len(indexed) != len(rows):
        raise ValueError("Empty evaluation or duplicate (task_id, seed)")
    return indexed


def paired_interval(left, right, metric, policy):
    # Shared bootstrap settings are supplied by retirement or attribution policy.
    a, b = _indexed(left), _indexed(right)
    if a.keys() != b.keys():
        raise ValueError("Paired evaluations must have identical task IDs and seeds")
    groups = {}
    for key in sorted(a):
        groups.setdefault(key[0], []).append(metric(a[key]) - metric(b[key]))
    # Equal weight to each task; all repeated seeds for a task form one cluster.
    differences = [mean(values) for values in groups.values()]
    rng = random.Random(policy.seed)
    boot = sorted(mean(rng.choices(differences, k=len(differences))) for _ in range(policy.bootstrap_samples))
    tail = (1 - policy.confidence) / 2
    return {"mean": mean(differences), "low": boot[int(tail * (len(boot) - 1))],
            "high": boot[int((1 - tail) * (len(boot) - 1))], "tasks": len(groups)}


def evaluate_model_acceptance(a, c, policy=RetirementPolicy()):
    """Reject clear full-Harness degradation, independently of module cost.

    Absence of significant degradation is not proof of non-inferiority.
    Insufficient independent tasks fail closed instead of accepting a model.
    """
    interval = paired_interval(c, a, lambda row: row.success, policy)
    checks = {
        "enough_independent_tasks": interval["tasks"] >= policy.min_tasks,
        "no_significant_full_degradation": interval["high"] >= -policy.performance_margin,
    }
    accepted = all(checks.values())
    reason = ("insufficient_independent_tasks" if not checks["enough_independent_tasks"] else
              "full_harness_degraded" if not checks["no_significant_full_degradation"] else
              "no_significant_full_degradation")
    return {"decision": "accept" if accepted else "rollback", "reason": reason,
            "checks": checks, "interval": interval, "policy": asdict(policy)}


def evaluate_retirement(cells: dict[str, list[EpisodeResult]], policy=RetirementPolicy()):
    if set(cells) != set("ABCD"):
        raise ValueError("All four cells A/B/C/D are required")
    keys = [_indexed(cells[cell]).keys() for cell in "ABCD"]
    if any(k != keys[0] for k in keys[1:]):
        raise ValueError("Four cells must share task IDs and seeds")
    score = lambda r: r.success
    intervals = {f"{a}-{b}": paired_interval(cells[a], cells[b], score, policy)
                 for a, b in (("A", "B"), ("D", "B"), ("D", "A"), ("D", "C"))}
    # Tokens cannot be replaced by module-count savings. Also require fewer
    # auxiliary calls and no increase in latency, tools or total model calls.
    cost_gates = {}
    for reference in ("A", "C"):
        for metric in ("total_tokens", "auxiliary_calls", "model_calls", "tool_calls", "latency_s"):
            getter = lambda r, m=metric: getattr(r.cost, m)
            interval = paired_interval(cells["D"], cells[reference], getter, policy)
            intervals[f"cost_{metric}_D-{reference}"] = interval
            baseline = mean(getter(r) for r in cells[reference])
            if metric == "total_tokens":
                passed = baseline > 0 and interval["high"] < -baseline * policy.min_token_saving_fraction
            elif metric == "auxiliary_calls":
                passed = interval["high"] < 0
            else:
                passed = interval["high"] <= 0
            cost_gates[f"{metric}_vs_{reference}"] = passed
    checks = {
        "enough_independent_tasks": intervals["A-B"]["tasks"] >= policy.min_tasks,
        "module_was_useful": intervals["A-B"]["low"] > 0,
        "student_improved_without_module": intervals["D-B"]["low"] > 0,
        "preserves_original_capability": intervals["D-A"]["low"] >= -policy.performance_margin,
        "small_remaining_dependence": intervals["D-C"]["low"] >= -policy.performance_margin,
        **cost_gates,
    }
    acceptance = evaluate_model_acceptance(cells["A"], cells["C"], policy)
    intervals["C-A"] = acceptance["interval"]
    capability_preserved = all(checks[key] for key in (
        "enough_independent_tasks", "module_was_useful", "student_improved_without_module",
        "preserves_original_capability", "small_remaining_dependence"))
    cost_improved = all(cost_gates.values())
    retirement_eligible = capability_preserved and cost_improved
    retirement_accepted = acceptance["decision"] == "accept" and retirement_eligible
    module_decision = "retire" if retirement_accepted else "retain"
    # `decision` is only a legacy module-decision alias, never model acceptance.
    return {"decision": module_decision, "module_decision": module_decision,
            "model_decision": acceptance["decision"], "model_acceptance": acceptance,
            "capability_preserved": capability_preserved, "cost_improved": cost_improved,
            "retirement_eligible": retirement_eligible, "retirement_accepted": retirement_accepted,
            "checks": checks, "intervals": intervals, "policy": asdict(policy),
            "success": {k: mean(r.success for r in v) for k, v in cells.items()}}


class PairedRetirementEvaluator:
    def __init__(self, runner, policy=RetirementPolicy()):
        self.runner, self.policy = runner, policy

    def evaluate(self, before_model, after_model, h_plus, h_minus, *, tasks, seeds, output, before_cells=None):
        from ..evolution.search import evaluate_tasks
        cells = dict(before_cells or {})
        for name, model, harness in (("A", before_model, h_plus), ("B", before_model, h_minus),
                                     ("C", after_model, h_plus), ("D", after_model, h_minus)):
            if name not in cells:
                result = evaluate_tasks(self.runner, model, harness, tasks, seeds, output / name)
                cells[name] = list(result.evaluations)
        return evaluate_retirement(cells, self.policy)
