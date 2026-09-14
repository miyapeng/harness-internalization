from __future__ import annotations

import json
import os
import time
import urllib.request
from dataclasses import asdict
from .candidate import Candidate
from ..core.types import Cost, write_json

CONTRACT = '''Propose executable control modules for agent harness optimization.
Return a JSON object with exactly one field "candidate_sources": a list of Python source strings.
Each source must contain literal assignments NAME, KIND, INSTRUCTION, PERSISTENCE,
and def trigger(history, step): return <expression>.
KIND is planner, review, or recovery; PERSISTENCE is an integer from 0 to 8.
Allowed expressions: history, step, strings/integers/bools, and/or/not,
single comparisons == != < <= > >= in not in, history.lower(), string.count(), len().
No imports, loops, I/O, environment mutations, API routines, answers, task IDs or hidden state.
Only generic control policies. Modules consume the same public history as the student.
The runtime implements planner -> final action, draft -> review -> final action,
or failure diagnosis -> final action, with no environment changes from internal calls.
Use a new NAME that does not collide with existing modules.
Candidate 1 and candidate 2 must address the same main failure with different control logic.
Use only the supplied search trajectories. Do not assume access to held-out tasks.
'''


class APIProposer:
    def __init__(self, *, model=None, base_url=None, api_key=None, transport=None, max_tokens=8192, temperature=0.7):
        self.model, self.base_url, self.api_key = model, base_url, api_key
        self.transport = transport
        self.max_tokens,self.temperature=max_tokens,temperature

    def request_json(self,contract,public,output):
        """Shared transport for code proposals; endpoint selection remains protected configuration."""
        payload={"model":self.model or os.environ["HI_PROPOSER_MODEL"],
            "messages":[{"role":"system","content":contract},{"role":"user","content":json.dumps(public,ensure_ascii=False)}],
            "max_tokens":self.max_tokens,"temperature":self.temperature,"response_format":{"type":"json_object"}}
        output.mkdir(parents=True,exist_ok=True)
        write_json(output/"proposer_prompt.json",payload)
        start=time.monotonic()
        if self.transport is not None: raw=self.transport(payload)
        else:
            url=(self.base_url or os.environ["HI_PROPOSER_BASE_URL"]).rstrip("/")+"/chat/completions"
            http=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={
                "Authorization":"Bearer "+(self.api_key or os.environ["HI_PROPOSER_API_KEY"]),"Content-Type":"application/json"})
            with urllib.request.urlopen(http,timeout=300) as response: raw=json.load(response)
        write_json(output/"proposer_response.json",raw)
        self.last_cost=Cost(raw["usage"]["prompt_tokens"],raw["usage"]["completion_tokens"],1,1,latency_s=time.monotonic()-start)
        write_json(output/"cost.json",asdict(self.last_cost))
        return json.loads(raw["choices"][0]["message"]["content"])

    def propose(self, request):
        allowed = set(request.tasks)
        if any(score.task_id not in allowed for score in request.scores):
            raise ValueError("Proposer scores outside search split")
        traces = []
        for trajectory in request.trajectories:
            if trajectory.task_id not in allowed:
                raise ValueError("Proposer trajectory outside search split")
            for step in trajectory.transitions:
                traces.append({"history": step.state.public_history, "step": step.state.step,
                               "action": step.action, "success": trajectory.success})
        payload = {"model": self.model or os.environ["HI_PROPOSER_MODEL"],
            "messages": [{"role": "system", "content": CONTRACT}, {"role": "user", "content": json.dumps({
                "candidate_count": request.count,
                "current_harness": {"version": request.harness.version,
                    "modules": [{"name": m.name, "source": m.source} for m in request.harness.modules]},
                "trajectories": traces[:32], "scores": [asdict(row) for row in request.scores],
                "candidate_history": request.history}, ensure_ascii=False)}],
            "max_tokens": 4096, "temperature": 0.7, "response_format": {"type": "json_object"}}
        request.output.mkdir(parents=True, exist_ok=True)
        write_json(request.output / "proposer_prompt.json", payload)
        started = time.monotonic()
        if self.transport is not None:
            raw = self.transport(payload)
        else:
            url = (self.base_url or os.environ["HI_PROPOSER_BASE_URL"]).rstrip("/") + "/chat/completions"
            key = self.api_key or os.environ["HI_PROPOSER_API_KEY"]
            http = urllib.request.Request(url, data=json.dumps(payload).encode(),
                headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
            with urllib.request.urlopen(http, timeout=300) as response: raw = json.load(response)
        write_json(request.output / "proposer_response.json", raw)
        sources = json.loads(raw["choices"][0]["message"]["content"])["candidate_sources"]
        if len(sources) != request.count or any(not isinstance(s, str) for s in sources):
            raise ValueError("Invalid proposer candidate count/types")
        usage = raw["usage"]
        self.last_cost = Cost(usage["prompt_tokens"], usage["completion_tokens"], 1, 1,
                              latency_s=time.monotonic() - started)
        candidates = tuple(Candidate(source, request.harness.version, request.cycle, i)
                           for i, source in enumerate(sources))
        for candidate in candidates:
            with (request.output / f"candidate_{candidate.index}.py").open("x") as file:
                file.write(candidate.source)
        return candidates
