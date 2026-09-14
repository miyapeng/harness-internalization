"""Terminal-Bench 2 via the official Harbor Trial lifecycle and a queue agent."""
from __future__ import annotations

import asyncio
from dataclasses import asdict
from importlib.metadata import version
import json
from pathlib import Path

from .common import Catalog, score01, tree_hash
from ..core.types import write_json
from ..training.rollout import EnvironmentStep

HARBOR_VERSION = "0.23.0"
STATUS = "implemented_mock_verified_harbor_docker_not_run"


class HarborSession:
    def __init__(self, config, output):
        self.config, self.output = config, Path(output)
        self.actions, self.observations = asyncio.Queue(), asyncio.Queue()
        self.task = None

    async def start(self, task_path):
        if version("harbor") != HARBOR_VERSION:
            raise RuntimeError(f"This bridge requires harbor=={HARBOR_VERSION}; revalidate before upgrading")
        from harbor.models.trial.config import TrialConfig
        from harbor.trial.trial import Trial
        from . import harbor_agent
        harbor_agent.SESSION = self
        config = TrialConfig.model_validate({"task":{"path":str(task_path)},
            "trial_name":"trial", "trials_dir":str(self.output/"harbor"),
            "agent":{"import_path":"internalization.benchmarks.harbor_agent:InternalizationAgent"},
            "environment":{"type":"docker", "delete":False}, "verifier":{"disable":False}})
        trial = await Trial.create(config)
        self.task = asyncio.create_task(trial.run())
        return await self.receive()

    async def receive(self):
        event = asyncio.create_task(self.observations.get())
        ready, _ = await asyncio.wait([event, self.task], return_when=asyncio.FIRST_COMPLETED)
        if event in ready: return event.result()
        event.cancel()
        await asyncio.gather(event, return_exceptions=True)
        result = self.task.result()
        if result.exception_info is not None: raise RuntimeError("Harbor trial failed; inspect private trial logs")
        if result.verifier_result is None or not result.verifier_result.rewards:
            raise RuntimeError("Harbor verifier did not return rewards")
        reward = score01(result.verifier_result.rewards["reward"])
        return {"terminal":True, "reward":reward}

    async def step(self, action, force_finish):
        await self.actions.put((action, force_finish))
        result = await self.receive()
        # Agent publishes its final command observation before returning to Harbor verification.
        if result.get("finish"):
            grade = await self.receive()
            if not grade.get("terminal"): raise RuntimeError("Unexpected Harbor lifecycle event")
            result.update(grade)
        return result

    async def close(self):
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


class TerminalBench2Environment:
    def __init__(self, config, output, *, training=False, session_factory=HarborSession):
        self.config, self.output, self.training = config, Path(output), training
        self.catalog = Catalog(config.catalog, "terminalbench2")
        self.session_factory, self.session, self.done = session_factory, None, True

    async def reset(self, task_id, seed):
        row = self.catalog.get(task_id, self.training)
        path = Path(row["task_path"]).resolve(strict=True)
        if tree_hash(path) != row["task_hash"]: raise ValueError("TB2 task bundle changed after import")
        self.task_id, self.seed, self.steps, self.done = task_id, seed, 0, False
        self.session = self.session_factory(self.config, self.output)
        first = await self.session.start(path)
        if "instruction" not in first: raise RuntimeError("Harbor agent did not start")
        self.history = [first["instruction"],
            'Use {"action":"exec","command":"..."} or {"action":"final"}. '
            'Commands run in the task container as its configured agent user. Files persist; each exec starts a shell.']
        write_json(self.output/"identity.json", {"task_id":task_id, "seed":seed,
            "catalog_hash":self.catalog.fingerprint, "task_hash":row["task_hash"],
            "harbor_version":HARBOR_VERSION, "seed_scope":"model replica; Harbor has no task seed parameter"})
        return {"observation":"\n".join(self.history)}

    async def step(self, action):
        if self.done: raise RuntimeError("Episode is already finished")
        self.steps += 1
        result = await self.session.step(action, self.steps >= self.config.max_steps)
        self.done = result.get("terminal", False)
        reward = score01(result["reward"]) if self.done else 0.
        self.history.extend(["Action: "+action, "Observation: "+result.get("observation", "Task submitted.")])
        if self.done:
            write_json(self.output/"grade.json", {"task_id":self.task_id, "seed":self.seed,
                "metrics":{"reward":reward}, "reward_metric":"harbor_verifier_reward"})
        return asdict(EnvironmentStep("\n".join(self.history), reward, self.done, reward,
            result.get("valid", True), result.get("tool_calls", 0), observation_kind="context"))

    async def close(self):
        if self.session is not None: await self.session.close()
