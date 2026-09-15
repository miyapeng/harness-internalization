"""Versioned source proposals over the existing configured proposer endpoint."""
from dataclasses import asdict

from .proposer import APITransport
from .candidate import HarnessCandidate
from ..harness.revision import InternalizationTarget
from ..core.types import Cost, write_json


def public_execution_trace(trajectory):
    trace={"task_id":trajectory.task_id,
        "steps":[{"context":s.student_prompt,"action":s.action} for s in trajectory.transitions],
        "score":trajectory.success,"total_reward":trajectory.total_reward}
    if hasattr(trajectory,"environment_events"):
        # Explicit public adapter results and submitted capability parameters only;
        # never serialize environment objects, evaluator state or hidden answers.
        trace["environment_events"]=[asdict(e) for e in trajectory.environment_events]
        trace["public_calls"]=[asdict(c) for c in trajectory.public_calls]
        trace["initial_observation"]=trajectory.initial_observation
    return trace

CONTRACT = '''Propose two Harness code patches, each addressing one main improvement.
Return {"candidates":[{"patch":[{"path":...,"content":...}],"rationale":...}]}.
Use only the supplied search information and current source. Paths are relative to the Harness
workspace. Each edit contains only path and content; the host binds it to the supplied parent.
content is the complete new file text, including for additions; content=null removes
the file from the NEW snapshot only. Multiple files and new tools are allowed.
Legacy config schema 1 has supervision: path.py:function or null. For named controls use
{"schema":2,"entrypoint":"agent/main.py:run","composition":"independent_suffix",
"controls":[{"id":"a_stable_behavior_id","entrypoint":"controls/example.py:run","enabled":true}]}.
IDs identify behaviors, not categories. Keep retained IDs, code and order unless intentionally
changing that behavior. Add new controls as separate IDs instead of merging everything into one hook.
All independent controls read the same base context and return suffix/selected; they cannot
read other control outputs, config/harness.json, or execute environment actions. Runtime combines
suffixes in list order. prepare/execute also cannot read config/harness.json in schema 2;
tool registrations and other shared files remain readable. A sequential_suffix composition may
read preceding suffixes and run environment actions, but is unsupported by the training bridge.
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

TARGET_CONTRACT = '''Select at most one behavior to attempt internalizing from the accepted Harness.
Return {"target":null} or
{"target":{"removed_behavior":"description of the behavior to bypass",
"supervision_adapter":"the supplied registered path.py:function"}}.
The current bridge can only bypass the registered supervision hook. The host will construct
the reduced revision by setting config.supervision to null, preserving all other config fields
and every other file, including tools, prompts and the hook source. Do not return patches,
file contents or hashes. A useful accepted Harness may have no supported target: return null.
The selected hook must do internal computation on the student's current public context only;
it must not request fresh environment observations, hidden answers or future information.
Executable compatibility checks still determine support; this selection is not proof of it.
Use only the supplied source and public search history; do not infer held-out task answers.
'''

NAMED_TARGET_CONTRACT = '''Select at most one enabled control ID from this accepted Harness.
Return {"target":null} or {"target":{"target_control_id":"registered ID","removed_behavior":"description"}}.
The host disables only that ID and preserves other controls, order, tools, files and config.
Do not return a patch, entrypoint, file contents or hashes. IDs are not control categories.
Only independent_suffix composition is supported for training: each hook reads the same base
public context without other control outputs or environment interaction. The host caches actual
non-target outputs and inserts the target at its registered position. Executable checks still
determine compatibility. Use only supplied source and public search history. Return null if no
behavior should be tried; a useful candidate need not be internalized.
'''


class CodeProposer(APITransport):
    def __init__(self,store,**kwargs):
        super().__init__(**kwargs)
        self.store=store

    def propose(self,request):
        allowed=set(request.tasks)
        if any(t.task_id not in allowed for t in request.trajectories) or any(s.task_id not in allowed for s in request.scores):
            raise ValueError("Proposer input outside search tasks")
        public={"candidate_count":request.count,"parent_revision":request.harness.version,
            "files":request.harness.files(),"workspace_policy":asdict(request.harness.policy),
            "traces":[public_execution_trace(t) for t in request.trajectories],
            "scores":[asdict(s) for s in request.scores],"history":request.history}
        raw=self.request_json(CONTRACT,public,request.output)
        proposals=raw["candidates"]
        if len(proposals)!=request.count: raise ValueError("Wrong code candidate count")
        results=[]
        for i,row in enumerate(proposals):
            try:
                patch=self.store.bind_patch(request.harness,row["patch"])
                candidate=HarnessCandidate.create(self.store,request.harness,patch,row["rationale"])
                write_json(request.output/f"candidate_{i}.json",candidate.to_dict())
                results.append(candidate)
            except Exception as exc:
                write_json(request.output/f"candidate_{i}_invalid.json",{"proposal":row,"reason":str(exc)})
                results.append({"invalid_proposal":row,"reason":str(exc)})
        return tuple(results)

    def propose_target(self,request):
        # No acceptance/retirement tasks, trajectories or gate outcomes are provided.
        if request.harness.config["schema"]==2: return self._propose_control_target(request)
        hook=request.harness.config["supervision"]
        if hook is None:
            self.last_cost=Cost()
            write_json(request.output/"cost.json",asdict(self.last_cost))
            write_json(request.output/"target_selection.json",{"target":None,"reason":"no_supervision_hook","model_called":False})
            return None
        raw=self.request_json(TARGET_CONTRACT,{"full_revision":request.harness.version,
            "files":request.harness.files(),"supervision_adapter":hook,"search_history":request.history},request.output)
        row=raw["target"]
        write_json(request.output/"target_selection.json",{"target":row,"model_called":True})
        if row is None: return None
        if not isinstance(row,dict) or set(row)!={"removed_behavior","supervision_adapter"}:
            raise ValueError("Target selection must contain only removed_behavior and supervision_adapter")
        target=InternalizationTarget.from_supervision(self.store,request.harness,
            row["removed_behavior"],row["supervision_adapter"])
        write_json(request.output/"internalization_target.json",target.to_dict())
        return target

    def _propose_control_target(self,request):
        config=request.harness.config
        controls=[c for c in config["controls"] if c["enabled"]]
        if config["composition"]!="independent_suffix" or not controls:
            reason="unsupported: dependent control composition" if controls else "no_enabled_controls"
            self.last_cost=Cost()
            write_json(request.output/"cost.json",asdict(self.last_cost))
            write_json(request.output/"target_selection.json",{"target":None,"reason":reason,"model_called":False})
            if controls: raise ValueError(reason)
            return None
        raw=self.request_json(NAMED_TARGET_CONTRACT,{"full_revision":request.harness.version,
            "files":request.harness.files(),"controls":controls,"search_history":request.history},request.output)
        row=raw["target"]
        write_json(request.output/"target_selection.json",{"target":row,"model_called":True})
        if row is None: return None
        if not isinstance(row,dict) or set(row)!={"target_control_id","removed_behavior"}:
            raise ValueError("Target selection must contain only target_control_id and removed_behavior")
        target=InternalizationTarget.from_control(self.store,request.harness,row["target_control_id"],row["removed_behavior"])
        write_json(request.output/"internalization_target.json",target.to_dict())
        return target
