"""Legacy project's all-in-one mock backend adapted to the independent interfaces."""
from .interfaces import Components
from .trajectory import RolloutResult, read_trace_file
from ..evolution.candidate import Candidate


def components_for(backend):
    if isinstance(backend, Components): return backend
    if hasattr(backend, "components"): return backend.components()

    class Proposer:
        def propose(self, request):
            paths = backend.propose(request.checkpoint, request.harness, request.tasks,
                                    request.cycle, request.count, request.output)
            return tuple(Candidate(path.read_text(), request.harness.version, request.cycle, i)
                         for i, path in enumerate(paths))

    class Runner:
        def rollout(self, model, harness, tasks, *, seeds, output, training=False):
            if training: raise ValueError("Legacy runner does not expose training trajectories")
            rows = backend.evaluate(model, harness, tasks, seeds, output)
            traces = read_trace_file(output / "trajectories.jsonl")
            return RolloutResult(traces, tuple(rows))

    class Trainer:
        def train(self, student, teacher, h_plus, h_minus, trajectories, *, tasks, target, budget, output):
            if student != teacher: raise ValueError("A phase starts with the same teacher/student snapshot")
            return backend.train(student, h_plus, h_minus, target, tasks, budget, output)

    return Components(Proposer(), Runner(), Trainer())
