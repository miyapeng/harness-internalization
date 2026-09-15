"""Synthetic lifecycle proof using actual sandboxed code; model/training decisions are mock."""
from dataclasses import asdict
import json
from pathlib import Path
import time

from .core.interfaces import Components
from .core.manifest import TaskManifest
from .core.types import Cost, write_json
from .evolution.candidate import HarnessCandidate
from .harness.revision import RevisionStore, FileEdit, InternalizationTarget, text_hash
from .harness.runtime import Completion
from .training.rollout import InteractionTaskRunner, EnvironmentStep
from .outer_loop import run_outer_loop, LoopConfig

ROOT=Path(__file__).resolve().parents[2]


def improvement(store,parent,*,diagnosis=True,unsupported=False):
    files=parent.files()
    additions=ROOT/"examples/versioned_harness/additions"
    registry={"log_query":{"entry":"tools.log_query:run","description":"Search log lines in the public history; argument needle:string"}}
    updates={"config/tools.json":json.dumps(registry),"tools/log_query.py":(additions/"tools/log_query.py").read_text()}
    if diagnosis:
        updates["controls/diagnosis.py"]=(additions/"controls/diagnosis.py").read_text()
        if unsupported:
            updates["controls/diagnosis.py"]='def augment(api, payload):\n    result=api.environment("observe")\n    return {"suffix":result["observation"],"selected":True}\n'
        updates["config/harness.json"]=json.dumps({**parent.config,"supervision":"controls/diagnosis.py:augment"})
    patch=tuple(FileEdit(p,text_hash(files[p]) if p in files else None,v) for p,v in updates.items())
    return HarnessCandidate.create(store,parent,patch,"Add public-log lookup and optional diagnostic control for recovery")


def reduction(store,full):
    config=json.dumps({**full.config,"supervision":None})
    reduced=store.apply(full,(FileEdit("config/harness.json",text_hash(full.files()["config/harness.json"]),config),))
    return InternalizationTarget(full,reduced,"Additional diagnostic control; retain the public-log query tool",full.config["supervision"])


class DemoEnvironment:
    def reset(self,task,seed): return "Synthetic task: finish after consulting the log.\nlog: action=finish"
    def step(self,action):
        if action=="observe": return EnvironmentStep("Observed public state.",0.,False,0.)
        return EnvironmentStep("Task ended.",float(action=="finish"),True,float(action=="finish"))
    def close(self): pass


class DemoModel:
    """Deterministic scripted behavior for acceptance cases, not a learned agent."""
    def __init__(self,checkpoint,*,requires_diagnosis=True):
        self.snapshot_id=str(checkpoint)
        self.config=json.loads((Path(checkpoint)/"mock_model.json").read_text())
        self.requires_diagnosis=requires_diagnosis
    def prompt_ids(self,prompt): return [ord(c)+1 for c in prompt]
    def generate(self,prompt,*,purpose):
        if purpose=="harness_internal":
            time.sleep(.02)  # Actual simulated auxiliary latency, not a fabricated measured number.
            return Completion("DIAGNOSIS: finish",Cost(len(prompt),20,1,1))
        if "[Registered tools]" in prompt and "LOG_QUERY_RESULT" not in prompt:
            text='{"tool":"log_query","arguments":{"needle":"action="}}'
        elif "LOG_QUERY_RESULT" in prompt and (not self.requires_diagnosis or self.config["trained"] or "DIAGNOSIS" in prompt):
            text="finish"
        else: text="fail"
        if self.config.get("broken"): text="fail"
        return Completion(text,Cost(len(prompt),len(text),1),tuple(ord(c)+1 for c in text))
    def score(self,prompt,ids):
        return [-.5 if "DIAGNOSIS" in prompt else -1.] * len(ids),Cost(len(prompt)+len(ids),0,1,1)


class DemoProposer:
    def __init__(self,store,scenario): self.store,self.scenario,self.requests=store,scenario,[]
    def propose(self,request):
        self.requests.append((request.checkpoint,request.harness.version))
        result=[]
        for i in range(request.count):
            candidate=improvement(self.store,request.harness,diagnosis=self.scenario!="tool_only",unsupported=self.scenario=="unsupported")
            if i:
                # Distinct executable candidate: same main purpose, a more explicit tool prompt.
                path="prompts/system.txt"
                files=request.harness.files()
                alternative=FileEdit(path,text_hash(files[path]),files[path]+"\nConsult the registered log lookup tool before acting.\n")
                candidate=HarnessCandidate.create(self.store,request.harness,(*candidate.patch,alternative),
                    "Add public-log lookup with explicit tool-use prompting and optional diagnostic control")
            if self.scenario!="tool_only":
                candidate=candidate.with_internalization(reduction(self.store,candidate.full_revision))
            write_json(request.output/f"candidate_{i}.json",candidate.to_dict())
            result.append(candidate)
        return tuple(result)


class DemoTrainer:
    def __init__(self,scenario): self.scenario,self.calls=scenario,[]
    def train(self,student,teacher,h_plus,h_minus,trajectories,*,tasks,target,budget,output):
        target.validate_structure()
        self.calls.append((h_plus.version,h_minus.version))
        checkpoint=output/"mock_checkpoint"
        write_json(checkpoint/"mock_model.json",{"trained":True,"broken":self.scenario=="rollback"})
        write_json(output/"mock_training.json",{"actual_optimizer_updates":0,"mock_checkpoint_transition":True,
            "allocated_budget":budget,"full_revision":h_plus.version,"reduced_revision":h_minus.version})
        return str(checkpoint.resolve())


def run_demo(output,scenario="mixed",config=LoopConfig()):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    store=RevisionStore(output/"revisions")
    initial=store.import_directory(ROOT/"examples/versioned_harness/base")
    checkpoint=output/"initial_model"
    write_json(checkpoint/"mock_model.json",{"trained":False,"broken":False})
    partitions={name:tuple(f"{name}-{i:03d}" for i in range(30)) for name in
        ("train","search","dev","test",*(f"retirement_{i}" for i in range(config.cycles)))}
    manifest=TaskManifest("synthetic_revision_lifecycle","fixture-v1",partitions)
    write_json(output/"manifest.json",asdict(manifest))
    proposer,trainer=DemoProposer(store,scenario),DemoTrainer(scenario)
    runner=InteractionTaskRunner(DemoEnvironment,lambda p:DemoModel(p,requires_diagnosis=scenario not in ("tool_only","attribution_failed","unsupported")),max_steps=3)
    result=run_outer_loop(Components(proposer,runner,trainer),manifest,str(checkpoint),output/"experiment",config,initial_harness=initial)
    write_json(output/"proof.json",{"scenario":scenario,"real_candidate_sandbox":True,"real_environment":False,
        "model_and_trainer":"scripted_mock","GPU":False,"actual_optimizer_updates":0,
        "trainer_calls":len(trainer.calls),"proposal_parents":proposer.requests,"deployment":result})
    return result
