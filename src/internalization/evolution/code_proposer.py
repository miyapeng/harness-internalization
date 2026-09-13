"""Versioned source proposals over the existing configured proposer endpoint."""
import json
from dataclasses import asdict

from .proposer import APIProposer
from .candidate import HarnessCandidate
from ..harness.revision import FileEdit, InternalizationTarget
from ..core.types import write_json

CONTRACT = '''Propose two Harness code patches, each addressing one main improvement.
Return {"candidates":[{"patch":[{"path":...,"before_hash":...,"content":...}],"rationale":...}]}.
Use only the supplied search information and current source. Paths are relative to the Harness
workspace. before_hash is SHA256 of current UTF-8 file, null for additions; content=null removes
the file from the NEW snapshot only. Multiple files and new tools are allowed. Keep config schema.
The protected runner calls entrypoint(api,payload): prepare returns prompt/tools/memory;
execute returns observation/memory/stop. Executable tools are dispatched by your entrypoint.
api.model(prompt) uses ONLY the current configured policy; api.environment(action) uses ONLY
the authorized task adapter. New tools may use these capabilities or public in-memory history.
No direct network, processes or filesystem outside the read-only revision/stdlib is permitted.
Never modify benchmark/evaluator/data partitions, reward/budget/safety code, weights/endpoints.
Optional supervision hooks run only internal calculations: (api,{context,step})->{suffix,selected}.
Do not propose task-specific answers or solutions. Planning/review/recovery are optional templates,
not validity categories. A useful candidate need not have an internalization target.
'''


class CodeProposer(APIProposer):
    def __init__(self,store,**kwargs):
        super().__init__(**kwargs)
        self.store=store

    def propose(self,request):
        allowed=set(request.tasks)
        if any(t.task_id not in allowed for t in request.trajectories) or any(s.task_id not in allowed for s in request.scores):
            raise ValueError("Proposer input outside search tasks")
        public={"candidate_count":request.count,"parent_revision":request.harness.version,
            "files":request.harness.files(),"workspace_policy":asdict(request.harness.policy),
            "traces":[{"task_id":t.task_id,"steps":[{"context":s.student_prompt,"action":s.action} for s in t.transitions],
                       "score":t.success} for t in request.trajectories],
            "scores":[asdict(s) for s in request.scores],"history":request.history}
        raw=self.request_json(CONTRACT,public,request.output)
        proposals=raw["candidates"]
        if len(proposals)!=request.count: raise ValueError("Wrong code candidate count")
        results=[]
        for i,row in enumerate(proposals):
            try:
                candidate=HarnessCandidate.create(self.store,request.harness,tuple(FileEdit(**p) for p in row["patch"]),row["rationale"])
                write_json(request.output/f"candidate_{i}.json",candidate.to_dict())
                results.append(candidate)
            except Exception as exc:
                write_json(request.output/f"candidate_{i}_invalid.json",{"proposal":row,"reason":str(exc)})
                results.append({"invalid_proposal":row,"reason":str(exc)})
        return tuple(results)

    def propose_target(self,request):
        # No acceptance/retirement tasks, trajectories or gate outcomes are provided.
        raw=self.request_json(CONTRACT+'''\nThe full candidate is already accepted. Return either
{"target":null} or {"target":{"patch":[...],"removed_behavior":"...","supervision_adapter":"path.py:function"}}.
For the current bridge, reduced code must preserve all files and bypass only config.supervision;
the hook stays in the source archive. Unsupported modifications should return null.
''',{"full_revision":request.harness.version,"files":request.harness.files(),"search_history":request.history},request.output)
        row=raw["target"]
        if row is None: return None
        reduced=self.store.apply(request.harness,tuple(FileEdit(**p) for p in row["patch"]))
        return InternalizationTarget(request.harness,reduced,row["removed_behavior"],row["supervision_adapter"])
