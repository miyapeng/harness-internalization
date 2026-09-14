"""Project-owned file/process boundary for isolated experiment components."""
from __future__ import annotations

import json
import subprocess
import sys
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
        self.sampling_state = getattr(self,"sampling_state",{})

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

        return Components(Proposer(), Runner(), Trainer(), execution_config=self.execution_config, sampling_state=self.sampling_state)

    def __init__(self, config: dict, ledger: Path):
        from .core.execution_config import resolve_execution
        allowed = {"propose", "target", "check_internalization", "evaluate", "train", "cwd", "timeout_s", "benchmark", "execution"}
        if set(config)-allowed: raise ValueError(f"Unknown backend fields: {sorted(set(config)-allowed)}")
        self.config = config
        self.sampling_state = {}
        limits = None
        # Resolve the benchmark's published limits once, then apply explicit experiment values.
        argv = config.get("evaluate", [])
        if isinstance(argv,list) and "--env-config" in argv:
            path = Path(argv[argv.index("--env-config")+1])
            if not path.is_absolute(): path = Path(config.get("cwd") or ".") / path
            benchmark = config.get("benchmark")
            if benchmark == "appworld":
                from .benchmarks.appworld import AppWorldConfig
                env = AppWorldConfig.load(path)
            elif benchmark in ("hotpotqa", "webshop", "terminalbench2", "swebench_pro", "lawbench"):
                from .benchmarks.common import BenchmarkConfig
                env = BenchmarkConfig.load(path)
            else: env = None
            if env is not None:
                limits = {"max_steps":env.max_steps, "model":{**env.model_options,"max_prompt_tokens":env.max_prompt_tokens}}
        self.execution_config = resolve_execution(config.get("execution"), benchmark_limits=limits)
        self.ledger = Journal(ledger)
        required = ("propose", "target", "check_internalization", "evaluate", "train") if self.execution_config["mode"] == "internalization" else ("propose", "evaluate")
        for key in required:
            if key not in config: raise ValueError(f"{self.execution_config['mode']} mode requires {key} entrypoint")
        for key in set(config) & {"propose", "target", "check_internalization", "evaluate", "train"}:
            command = config[key]
            if not isinstance(command, list) or not command or any(not isinstance(s, str) for s in command):
                raise ValueError("Commands must be nonempty argv arrays; shell strings are not allowed")
        if type(config.get("timeout_s",86400)) not in (int,float) or config.get("timeout_s",86400) <= 0:
            raise ValueError("Positive process timeout required")

    def _call(self, stage, payload, output):
        output.mkdir(parents=True, exist_ok=True)
        request, response = output / "request.json", output / "response.json"
        from .core.execution_config import config_hash, save_effective
        save_effective(output, self.execution_config)
        write_json(request, {"schema_version": 1, "stage": stage, **payload,
            "effective_config": self.execution_config, "effective_config_hash": config_hash(self.execution_config)})
        if response.exists(): raise FileExistsError(response)
        argv = [s.replace("{request}", str(request.resolve())).replace("{response}", str(response.resolve())).replace("{python}",sys.executable)
                for s in self.config[stage]]
        start = time.monotonic()
        with (output / "stdout.log").open("x") as stdout, (output / "stderr.log").open("x") as stderr:
            import os
            env=dict(os.environ, PYTHONHASHSEED=str(self.execution_config["seeds"]["model_sampling_seed"]))
            try:
                completed = subprocess.run(argv, cwd=self.config.get("cwd"), env=env, shell=False,
                             timeout=self.config.get("timeout_s", 86400), stdout=stdout, stderr=stderr)
            except subprocess.TimeoutExpired:
                self.ledger.append("process_timeout",stage=stage,request=str(request),wall_time_s=time.monotonic()-start)
                raise
            finally:
                # The child is reaped on timeout. Preserve consumption even if no response was produced.
                if stage == "train" and self.execution_config["schedule"]["profile"]=="budget_v1":
                    cursor=output/"sampling_state.json"
                    if cursor.exists():
                        from .core.sampling import TaskQueue
                        state=json.loads(cursor.read_text())
                        TaskQueue(payload["task_ids"],self.execution_config["seeds"]["run_seed"],state)
                        self.sampling_state.clear(); self.sampling_state.update(state)
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
            "task_ids": tasks, "planned_update_batches": budget,
            **({"sampling_state":self.sampling_state} if self.execution_config["schedule"]["profile"]=="budget_v1" else {})}, output)
        if self.execution_config["schedule"]["profile"] == "budget_v1":
            from .core.sampling import TaskQueue
            state = result.get("sampling_state")
            if not state: raise ValueError("Trainer did not return its sampling cursor")
            TaskQueue(tasks,self.execution_config["seeds"]["run_seed"],state)
            self.sampling_state.clear()
            self.sampling_state.update(state)
        if any(type(result.get(k)) is not int or result[k] != budget for k in ("planned_update_batches", "attempted_update_batches")):
            raise ValueError("Training did not consume the prescribed update-batch budget")
        calls = result.get("actor_update_calls")
        if type(calls) is not int or not 0 <= calls <= budget: raise ValueError("Invalid actor update count")
        if calls == 0:
            from .core.execution_config import NoActorUpdates
            if result.get("status") != "no_actor_updates" or result.get("checkpoint") != checkpoint:
                raise ValueError("Zero updates must preserve the input checkpoint")
            raise NoActorUpdates(result)
        if result.get("status") != "trained": raise ValueError("Actor calls do not imply a successfully trained model")
        steps=result.get("optimizer_steps")
        if steps is not None and (type(steps) is not int or steps < 0): raise ValueError("Invalid actual optimizer step count")
        path = Path(result["checkpoint"])
        if not path.is_absolute() or not path.is_dir():
            raise ValueError("Trainer must return an existing absolute checkpoint directory")
        return str(path)
