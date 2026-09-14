from __future__ import annotations

from dataclasses import dataclass
from .types import Cost, EpisodeResult, State


@dataclass(frozen=True)
class Transition:
    state: State
    student_prompt: str
    action: str
    response_ids: tuple[int, ...] = ()
    reward: float = 0.0
    done: bool = False
    action_valid: bool = True
    old_log_probs: tuple[float, ...] = ()
    cost: Cost = Cost()
    prompt_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class ControlOutput:
    control_id: str
    entrypoint: str
    suffix: str
    selected: bool


@dataclass(frozen=True)
class ControlContext:
    base_context: str
    composition: str
    outputs: tuple[ControlOutput, ...]


@dataclass(frozen=True)
class RevisionTransition(Transition):
    # Only schema-2 traces need composition metadata; old serialized traces remain unchanged.
    control_context: ControlContext | None = None


@dataclass(frozen=True)
class Trajectory:
    task_id: str
    episode_id: str
    seed: int
    model_version: str
    harness_version: str
    transitions: tuple[Transition, ...]
    success: float
    cost: Cost
    group_id: str = ""

    def __post_init__(self):
        for index, transition in enumerate(self.transitions):
            if transition.state.task_id != self.task_id or transition.state.episode_id != self.episode_id:
                raise ValueError("Trajectory identity mismatch")
            if transition.state.step != index:
                raise ValueError("Trajectory steps must be contiguous")

    @property
    def outcome(self):
        return EpisodeResult(self.task_id, self.seed, self.success, self.cost)

    @property
    def total_reward(self):
        return sum(step.reward for step in self.transitions)


@dataclass(frozen=True)
class EnvironmentEvent:
    index: int
    step: int
    phase: str
    action: str
    observation: str
    reward: float
    done: bool
    success: float
    action_valid: bool
    tool_calls: int
    control_id: str | None = None


@dataclass(frozen=True)
class RuntimeCall:
    index: int
    step: int
    phase: str
    operation: str
    parameters: dict
    result: str | dict
    cost: Cost
    control_id: str | None = None


@dataclass(frozen=True)
class EventTrajectory(Trajectory):
    """Complete tool returns independent of whether a student decision was generated.

    An explicitly empty event ledger means zero environment reward. It must never
    fall back to transition rewards or add them again. Old Trajectory JSON stays valid.
    """
    environment_events: tuple[EnvironmentEvent, ...] = ()
    public_calls: tuple[RuntimeCall, ...] = ()
    initial_observation: str = ""

    def __post_init__(self):
        super().__post_init__()
        for records in (self.environment_events,self.public_calls):
            for index,record in enumerate(records):
                if record.index!=index or not 0<=record.step<=len(self.transitions):
                    raise ValueError("Runtime event index/decision step mismatch")
        if any(event.done for event in self.environment_events[:-1]):
            raise ValueError("Environment event recorded after task termination")

    @property
    def total_reward(self):
        return sum(event.reward for event in self.environment_events)


@dataclass(frozen=True)
class RolloutResult:
    trajectories: tuple[Trajectory, ...]
    evaluations: tuple[EpisodeResult, ...]


def read_trace_file(path):
    """Import the previous project's public JSONL trace format, without labels."""
    import json
    from collections import defaultdict
    groups = defaultdict(list)
    if not path.exists(): return ()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    complete = [row["trajectory"] for row in rows if row.get("kind") == "trajectory"]
    if complete:
        from .serialization import trajectory_from_dict
        return tuple(trajectory_from_dict(value) for value in complete)
    for row in rows:
        if "state" not in row or "action" not in row: continue
        state = State(**row["state"])
        groups[(state.task_id, state.episode_id)].append((state, row))
    trajectories = []
    for (task_id, episode_id), records in groups.items():
        records.sort(key=lambda record: record[0].step)
        steps = tuple(Transition(state, state.public_history, row["action"],
            reward=float(row.get("reward", 0)), done=bool(row.get("done", False))) for state, row in records)
        trajectories.append(Trajectory(task_id, episode_id, 0, "legacy", "legacy", steps,
            float(records[-1][1].get("success", 0)), Cost()))
    return tuple(trajectories)
