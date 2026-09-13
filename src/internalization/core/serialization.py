from .types import Cost, EpisodeResult, State
from .trajectory import Transition, Trajectory
from ..harness.module import Harness, HarnessModule


def harness_from_dict(value):
    if value.get("format")=="code_revision_v1":
        from ..harness.revision import HarnessRevision
        return HarnessRevision.from_dict(value)
    harness = Harness(tuple(HarnessModule.from_source(m["source"]) for m in value["modules"]))
    if harness.version != value["version"]: raise ValueError("Harness hash mismatch")
    return harness


def trajectory_from_dict(value):
    transitions = []
    for row in value["transitions"]:
        row = dict(row)
        row["state"], row["cost"] = State(**row["state"]), Cost(**row["cost"])
        for key in ("response_ids", "old_log_probs", "prompt_ids"):
            row[key] = tuple(row.get(key, ()))
        transitions.append(Transition(**row))
    return Trajectory(**{**value, "transitions": tuple(transitions), "cost": Cost(**value["cost"])})
