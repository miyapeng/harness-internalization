from .types import Cost, EpisodeResult, State
from .trajectory import (Transition, Trajectory, RevisionTransition, ControlContext, ControlOutput,
                         EventTrajectory, EnvironmentEvent, RuntimeCall)
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
        if "control_context" in row:
            context=row["control_context"]
            if context is not None:
                row["control_context"]=ControlContext(context["base_context"],context["composition"],
                    tuple(ControlOutput(**o) for o in context["outputs"]))
            transitions.append(RevisionTransition(**row))
        else: transitions.append(Transition(**row))
    fields={**value, "transitions": tuple(transitions), "cost": Cost(**value["cost"])}
    if "environment_events" in value:
        fields["environment_events"]=tuple(EnvironmentEvent(**e) for e in value["environment_events"])
        fields["public_calls"]=tuple(RuntimeCall(**{**c,"cost":Cost(**c["cost"])}) for c in value.get("public_calls",()))
        return EventTrajectory(**fields)
    return Trajectory(**fields)
