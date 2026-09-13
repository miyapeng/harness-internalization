"""Execute a revision's prepare/dispatch code through a protected capability broker."""
from dataclasses import asdict
import json

from .sandbox import SandboxedCode
from ..core.types import Cost
from ..training.rollout import EnvironmentStep


class CapabilityBroker:
    def __init__(self,model,revision,*,environment=None,audit=None):
        self.model,self.revision,self.environment,self.audit=model,revision,environment,audit
        self.cost,self.reward,self.done,self.success,self.public=Cost(),0.,False,0.,None
        self.valid=True

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
        else: raise ValueError("No broker for requested capability")
        if self.audit: self.audit.append("capability",operation=operation,revision=self.revision.version,
            cost={k:v-before[k] for k,v in asdict(self.cost).items()})
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
    def __init__(self,model,revision,environment,*,sandbox=None,audit=None):
        self.model,self.revision,self.environment=model,revision,environment
        self.sandbox,self.audit=sandbox or SandboxedCode(),audit
        self.memory={}
        self.tools={}

    def prepare(self,history,step):
        self.broker=CapabilityBroker(self.model,self.revision,environment=self.environment,audit=self.audit)
        result=self.sandbox.call(self.revision,self.revision.config["entrypoint"],
            {"operation":"prepare","history":history,"step":step,"memory":self.memory},self.broker,audit=self.audit)
        if not isinstance(result,dict) or set(result)!={"prompt","tools","memory"}:
            raise ValueError("Harness prepare must return prompt, tools and memory")
        if not isinstance(result["prompt"],str) or not isinstance(result["tools"],dict) or not isinstance(result["memory"],dict):
            raise ValueError("Invalid prepare response")
        self.memory,self.tools=result["memory"],result["tools"]
        # Tool registration is executable candidate output, included in the actual action prompt.
        context=result["prompt"]
        if self.tools: context+="\n[Registered tools]\n"+json.dumps(self.tools,ensure_ascii=False,sort_keys=True)
        enhanced,_,_=augment_context(self.model,self.revision,context,step,sandbox=self.sandbox,audit=self.audit,broker=self.broker)
        return enhanced,self.broker.done

    def execute(self,action,history,step):
        result=self.sandbox.call(self.revision,self.revision.config["entrypoint"],
            {"operation":"execute","action":action,"history":history,"step":step,
             "memory":self.memory,"tools":self.tools},self.broker,audit=self.audit)
        if (not isinstance(result,dict) or set(result)!={"observation","memory","stop"} or
            not isinstance(result["observation"],str) or not isinstance(result["memory"],dict) or type(result["stop"]) is not bool):
            raise ValueError("Harness execute must return observation, memory and stop; reward stays protected")
        self.memory=result["memory"]
        # Preserve actual tool observations even if the candidate omits them from its return value.
        public=history
        if self.broker.public is not None:
            public+="\n[Environment observation]\n"+self.broker.public
        public+="\n[Student action]\n"+action+"\n[Harness tool result]\n"+result["observation"]
        if self.broker.public is None: self.broker.cost+=Cost(tool_calls=1)  # One local registered-tool dispatch.
        return EnvironmentStep(public,self.broker.reward,self.broker.done or result["stop"],self.broker.success,self.broker.valid)
