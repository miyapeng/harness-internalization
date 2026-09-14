"""Project-owned file/process boundary for isolated experiment components."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from dataclasses import asdict
from .core.types import Cost, EpisodeResult, Journal, write_json


def serialize_harness(harness):
    from .core.accepted_state import serialize_harness as serialize
    return serialize(harness)


class CommandBackend:
    def components(self):
        from .core.interfaces import Components
        from .core.trajectory import RolloutResult, read_trace_file
        from .evolution.candidate import Candidate
        owner = self

        class Proposer:
            def propose(self, request):
                result = owner._call("propose", {"checkpoint": request.checkpoint,
                    "harness": serialize_harness(request.harness), "task_ids": request.tasks,
                    "cycle": request.cycle, "candidate_count": request.count,
                    "trajectories": [asdict(t) for t in request.trajectories],
                    "scores": [asdict(r) for r in request.scores], "history": request.history}, request.output)
                if "candidates" in result:
                    from .evolution.candidate import HarnessCandidate
                    return tuple(HarnessCandidate.from_dict(row) if "candidate_id" in row else row for row in result["candidates"])
                return tuple(Candidate(source, request.harness.version, request.cycle, i)
                             for i, source in enumerate(result["candidate_sources"]))

            def propose_target(self,request):
                if "target" not in owner.config: return None
                from .harness.revision import InternalizationTarget
                result=owner._call("target",{"checkpoint":request.checkpoint,"harness":serialize_harness(request.harness),
                    "task_ids":request.tasks,"cycle":request.cycle,"candidate_count":1,
                    "trajectories":[],"scores":[],"history":request.history},request.output)
                return InternalizationTarget.from_dict(result["target"]) if result["target"] is not None else None

        class Runner:
            def rollout(self, model, harness, tasks, *, seeds, output, training=False):
                if training: raise ValueError("Command trainer owns its fresh rollout collection")
                rows = owner.evaluate(model, harness, tasks, seeds, output)
                return RolloutResult(read_trace_file(output / "trajectories.jsonl"), tuple(rows))

            def check_internalization(self,model,target,tasks,*,output):
                if "check_internalization" not in owner.config:
                    return {"supported":False,"reason":"unsupported: command backend has no executable target checker"}
                return owner._call("check_internalization",{"checkpoint":model,"target":target.to_dict(),
                    "task_ids":tasks},output)

        class Trainer:
            def train(self, student, teacher, h_plus, h_minus, trajectories, *, tasks, target, budget, output):
                if student != teacher: raise ValueError("Teacher must be the phase-initial student snapshot")
                return owner.train(student, h_plus, h_minus, target, tasks, budget, output)

        return Components(Proposer(), Runner(), Trainer())

    def __init__(self, config: dict, ledger: Path):
        self.config = config
        self.ledger = Journal(ledger)
        for key in ("propose", "evaluate", "train"):
            command = config[key]
            if not isinstance(command, list) or not command or any(not isinstance(s, str) for s in command):
                raise ValueError("Commands must be nonempty argv arrays; shell strings are not allowed")

    def _call(self, stage, payload, output):
        output.mkdir(parents=True, exist_ok=True)
        request, response = output / "request.json", output / "response.json"
        write_json(request, {"schema_version": 1, "stage": stage, **payload})
        if response.exists(): raise FileExistsError(response)
        argv = [s.replace("{request}", str(request.resolve())).replace("{response}", str(response.resolve()))
                for s in self.config[stage]]
        start = time.monotonic()
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            completed = subprocess.run(argv, cwd=self.config.get("cwd"), shell=False,
                         timeout=self.config.get("timeout_s", 86400), stdout=stdout, stderr=stderr)
        elapsed = time.monotonic() - start
        self.ledger.append("process", stage=stage, request=str(request), returncode=completed.returncode,
                           wall_time_s=elapsed)
        completed.check_returncode()
        result = json.loads(response.read_text())
        if result.get("schema_version") != 1:
            raise ValueError("Unsupported response schema")
        # Every process must report its full aggregate consumption, even if 0.
        required = {"input_tokens", "output_tokens", "model_calls", "auxiliary_calls", "tool_calls", "latency_s"}
        if not required.issubset(result.get("cost", {})):
            raise ValueError("Stage response lacks complete cost accounting")
        cost = Cost(**result["cost"])
        self.ledger.append("stage_cost", stage=stage, cost=result["cost"], wall_time_s=elapsed)
        return result

    def propose(self, checkpoint, harness, tasks, cycle, count, output):
        from .core.interfaces import ProposalRequest
        from .core.trajectory import read_trace_file
        from .core.types import read_results
        baseline = output.parent / "baseline_search"
        scores = read_results(baseline / "episodes.json")
        request = ProposalRequest(checkpoint, harness, tuple(tasks),
            read_trace_file(baseline / "trajectories.jsonl"), tuple(scores), (), cycle, count, output)
        candidates = self.components().proposer.propose(request)
        paths = []
        for i, candidate in enumerate(candidates):
            path = output / f"candidate_{i}.py"
            if path.exists():
                if path.read_text() != candidate.source: raise ValueError("Candidate source mismatch")
            else:
                with path.open("x") as f: f.write(candidate.source)
            paths.append(path)
        return paths

    def evaluate(self, checkpoint, harness, tasks, seeds, output):
        result = self._call("evaluate", {"checkpoint": checkpoint, "harness": serialize_harness(harness),
            "task_ids": tasks, "seeds": seeds}, output)
        return [EpisodeResult(r["task_id"], r["seed"], r["success"], Cost(**r["cost"])) for r in result["episodes"]]

    def train(self, checkpoint, full, reduced, target, tasks, budget, output):
        from .harness.revision import InternalizationTarget
        result = self._call("train", {"teacher_checkpoint": checkpoint, "student_checkpoint": checkpoint,
            "full_harness": serialize_harness(full), "reduced_harness": serialize_harness(reduced),
            "target": target.to_dict() if isinstance(target,InternalizationTarget) else target,
            "task_ids": tasks, "optimizer_steps": budget}, output)
        if result.get("optimizer_steps_completed") != budget:
            raise ValueError("Training did not consume the prescribed update-batch budget")
        path = Path(result["checkpoint"])
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("Trainer must return an existing absolute checkpoint directory")
        return str(path)
