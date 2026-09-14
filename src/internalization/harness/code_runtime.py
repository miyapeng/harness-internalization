"""Execute a revision's prepare/dispatch code through a protected capability broker."""
from dataclasses import asdict
import json

from .sandbox import SandboxedCode
from ..core.types import Cost
from ..core.trajectory import EnvironmentEvent, RuntimeCall
from ..training.rollout import EnvironmentStep


class CapabilityBroker:
    def __init__(self,model,revision,*,environment=None,audit=None,events=None,calls=None,context=None):
        self.model,self.revision,self.environment,self.audit=model,revision,environment,audit
        self.cost,self.reward,self.done,self.success,self.public=Cost(),0.,False,0.,None
        self.valid=True
        self.environment_events=events if events is not None else []
        self.public_calls=calls if calls is not None else []
        self.context=dict(context or {"step":0,"phase":"internal"})

    def record_call(self,operation,parameters,result,cost):
        call=RuntimeCall(len(self.public_calls),self.context["step"],self.context["phase"],
            operation,dict(parameters),result,cost,self.context.get("control_id"))
        self.public_calls.append(call)
        if self.audit: self.audit.append("capability",revision=self.revision.version,
            task_id=self.context.get("task_id"),episode_id=self.context.get("episode_id"),
            model_version=self.model.snapshot_id,**asdict(call))

    def __call__(self,operation,payload):
        before=asdict(self.cost)
        if operation not in self.revision.policy.capabilities: raise ValueError("Unauthorized candidate capability")
        if operation=="model":
            if set(payload)!={"prompt"} or not isinstance(payload["prompt"],str): raise ValueError("Invalid model capability")
            response=self.model.generate(payload["prompt"],purpose="harness_internal")
            self.cost+=response.cost
            result=response.text
        elif operation=="environment":
            if self.environment is None: raise ValueError("unsupported: supervision attempted new environment observation/effect")
            if self.done: raise ValueError("Environment already terminated")
            if set(payload)!={"action"} or not isinstance(payload["action"],str): raise ValueError("Invalid environment capability")
            outcome=self.environment.step(payload["action"])
            self.cost+=Cost(tool_calls=outcome.tool_calls)
            self.reward+=outcome.reward
            self.valid=self.valid and outcome.action_valid
            self.done,self.success,self.public=outcome.done,outcome.success,outcome.observation
            result={"observation":outcome.observation,"done":outcome.done,"action_valid":outcome.action_valid}
            event=EnvironmentEvent(len(self.environment_events),self.context["step"],self.context["phase"],
                payload["action"],outcome.observation,outcome.reward,outcome.done,outcome.success,
                outcome.action_valid,outcome.tool_calls,self.context.get("control_id"),outcome.observation_kind)
            self.environment_events.append(event)
            if self.audit: self.audit.append("environment_event",revision=self.revision.version,
                task_id=self.context.get("task_id"),episode_id=self.context.get("episode_id"),
                model_version=self.model.snapshot_id,**asdict(event))
        else: raise ValueError("No broker for requested capability")
        self.record_call(operation,payload,result,Cost(**{k:v-before[k] for k,v in asdict(self.cost).items()}))
        return result


def augment_context(model,revision,context,step,*,sandbox=None,audit=None,broker=None):
    entrypoint=revision.config["supervision"]
    if entrypoint is None: return context,False,Cost()
    broker=broker or CapabilityBroker(model,revision,audit=audit)  # Scoring defaults to no environment/hidden state.
    before=asdict(broker.cost)
    result=(sandbox or SandboxedCode()).call(revision,entrypoint,{"context":context,"step":step},broker,audit=audit)
    if (not isinstance(result,dict) or set(result)!={"suffix","selected"} or
        not isinstance(result["suffix"],str) or type(result["selected"]) is not bool):
        raise ValueError("unsupported: supervision must return executable suffix/selected contract")
    if not result["selected"] and result["suffix"]:
        raise ValueError("unsupported: inactive supervision cannot change the student context")
    # Never regenerate unrelated context, memory or non-target model guidance.
    used=Cost(**{k:v-before[k] for k,v in asdict(broker.cost).items()})
    return context+result["suffix"],result["selected"],used


class CodeRuntime:
    def __init__(self,model,revision,environment,*,sandbox=None,audit=None,task_id=None,episode_id=None):
        self.model,self.revision,self.environment=model,revision,environment
        self.sandbox,self.audit=sandbox or SandboxedCode(),audit
        self.memory={}
        self.tools={}
        self.environment_events=[]
        self.public_calls=[]
        self.task_id,self.episode_id=task_id,episode_id
        self.environment_history=None
        self.local_history=[]
        self.rendered_events=0

    def prepare(self,history,step):
        if self.environment_history is None: self.environment_history=history
        named=self.revision.config["schema"]==2
        self.broker=CapabilityBroker(self.model,self.revision,environment=self.environment,audit=self.audit,
            events=self.environment_events,calls=self.public_calls,
            context={"task_id":self.task_id,"episode_id":self.episode_id,"step":step,"phase":"prepare"})
        result=self.sandbox.call(self.revision,self.revision.config["entrypoint"],
            {"operation":"prepare","history":history,"step":step,"memory":self.memory},self.broker,audit=self.audit,
            **({"hide_control_config":True} if named else {}))
        if not isinstance(result,dict) or set(result)!={"prompt","tools","memory"}:
            raise ValueError("Harness prepare must return prompt, tools and memory")
        if not isinstance(result["prompt"],str) or not isinstance(result["tools"],dict) or not isinstance(result["memory"],dict):
            raise ValueError("Invalid prepare response")
        self.memory,self.tools=result["memory"],result["tools"]
        # Tool registration is executable candidate output, included in the actual action prompt.
        context=result["prompt"]
        if self.tools: context+="\n[Registered tools]\n"+json.dumps(self.tools,ensure_ascii=False,sort_keys=True)
        self.control_context=None
        self.broker.context["phase"]="control"
        if self.broker.done: return context,True  # No control/model work after environment termination.
        if named:
            from .control_runtime import run_controls
            enhanced,self.control_context,used=run_controls(self.model,self.revision,context,step,
                sandbox=self.sandbox,audit=self.audit,broker=self.broker)
            if self.revision.config["composition"]=="independent_suffix": self.broker.cost+=used
        else:
            enhanced,_,_=augment_context(self.model,self.revision,context,step,sandbox=self.sandbox,audit=self.audit,broker=self.broker)
        return enhanced,self.broker.done

    def execute(self,action,history,step):
        self.broker.context["phase"]="execute"
        self.broker.context.pop("control_id",None)
        previous_events=len(self.environment_events)
        result=self.sandbox.call(self.revision,self.revision.config["entrypoint"],
            {"operation":"execute","action":action,"history":history,"step":step,
             "memory":self.memory,"tools":self.tools},self.broker,audit=self.audit,
            **({"hide_control_config":True} if self.revision.config["schema"]==2 else {}))
        if (not isinstance(result,dict) or set(result)!={"observation","memory","stop"} or
            not isinstance(result["observation"],str) or not isinstance(result["memory"],dict) or type(result["stop"]) is not bool):
            raise ValueError("Harness execute must return observation, memory and stop; reward stays protected")
        self.memory=result["memory"]
        # Render event occurrences, not distinct strings. A full environment context
        # replaces its previous view. Local tool history has its own retained view.
        fresh=self.environment_events[self.rendered_events:]
        for event in fresh:
            if event.index == previous_events:
                self.environment_history+="\n[Student action]\n"+action
            if event.observation_kind=="context":
                self.environment_history=event.observation
            else:
                self.environment_history+="\n[Environment observation]\n"+event.observation
        self.rendered_events=len(self.environment_events)
        # A dispatcher commonly echoes the exact environment result. Suppress that
        # mirror only against events in THIS dispatch, never against previous text.
        mirrors_event=any(result["observation"]==event.observation for event in fresh)
        if not mirrors_event:
            self.local_history.append(("\n[Student action]\n"+action if len(self.environment_events)==previous_events else "")
                +"\n[Harness tool result]\n"+result["observation"])
        public=self.environment_history+"".join(self.local_history)
        if len(self.environment_events)==previous_events:
            used=Cost(tool_calls=1)  # One local dispatch, even if prepare previously called the environment.
            self.broker.cost+=used
            self.broker.record_call("local_tool",{"action":action},
                {"observation":result["observation"],"stop":result["stop"]},used)
        return EnvironmentStep(public,self.broker.reward,self.broker.done or result["stop"],self.broker.success,self.broker.valid,observation_kind="context")
