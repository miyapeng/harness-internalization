"""Named controls: explicit composition, with replay of non-target rollout outputs."""
from dataclasses import asdict

from .code_runtime import CapabilityBroker
from .sandbox import SandboxedCode
from ..core.trajectory import ControlContext, ControlOutput
from ..core.types import Cost


def compose_context(base,outputs):
    return base+"".join(o.suffix for o in outputs)


def execute_control(model,revision,control,context,step,*,sandbox=None,audit=None,broker=None):
    broker=broker or CapabilityBroker(model,revision,audit=audit,context={"step":step,"phase":"control"})
    broker.context["control_id"]=control["id"]
    before=asdict(broker.cost)
    result=(sandbox or SandboxedCode()).call(revision,control["entrypoint"],
        {"context":context,"step":step},broker,audit=audit,hide_control_config=True)
    if (not isinstance(result,dict) or set(result)!={"suffix","selected"} or
        not isinstance(result["suffix"],str) or type(result["selected"]) is not bool):
        raise ValueError("unsupported: control must return suffix/selected")
    if not result["selected"] and result["suffix"]:
        raise ValueError("unsupported: inactive control cannot change context")
    output=ControlOutput(control["id"],control["entrypoint"],result["suffix"],result["selected"])
    used=Cost(**{k:v-before[k] for k,v in asdict(broker.cost).items()})
    if audit: audit.append("named_control",revision=revision.version,step=step,output=asdict(output),cost=asdict(used))
    return output,used


def run_controls(model,revision,base,step,*,sandbox=None,audit=None,broker=None):
    config=revision.config
    outputs=[];cost=Cost()
    for control in config["controls"]:
        if not control["enabled"]: continue
        independent=config["composition"]=="independent_suffix"
        context=base if independent else compose_context(base,outputs)
        # Independent mode has no shared mutable broker/environment or other control outputs.
        selected_broker=broker
        if independent:
            selected_broker=CapabilityBroker(model,revision,audit=audit,
                events=broker.environment_events if broker else None,calls=broker.public_calls if broker else None,
                context=broker.context if broker else {"step":step,"phase":"control"})
        output,used=execute_control(model,revision,control,context,step,sandbox=sandbox,audit=audit,broker=selected_broker)
        outputs.append(output);cost+=used
    trace=ControlContext(base,config["composition"],tuple(outputs))
    return compose_context(base,outputs),trace,cost


def score_control_context(model,target,transition,*,audit=None):
    trace=getattr(transition,"control_context",None)
    if not isinstance(trace,ControlContext) or trace.composition!="independent_suffix":
        raise ValueError("unsupported: missing independent control rollout context")
    full=target.full_revision.config
    expected=[c for c in target.reduced_revision.config["controls"] if c["enabled"]]
    if [(o.control_id,o.entrypoint) for o in trace.outputs]!=[(c["id"],c["entrypoint"]) for c in expected]:
        raise ValueError("Control rollout IDs/order/entrypoints do not match H-minus")
    for output in trace.outputs:
        if (not isinstance(output.suffix,str) or type(output.selected) is not bool or
            (not output.selected and output.suffix)):
            raise ValueError("Invalid retained control output")
    if not isinstance(trace.base_context,str) or compose_context(trace.base_context,trace.outputs)!=transition.student_prompt:
        raise ValueError("Control composition does not reproduce actual student prompt")
    control=next(c for c in full["controls"] if c["id"]==target.target_control_id)
    output,cost=execute_control(model,target.full_revision,control,trace.base_context,transition.state.step,audit=audit)
    cached={o.control_id:o for o in trace.outputs}
    cached[control["id"]]=output
    # Insert at the deployment position, even if the target precedes retained controls.
    ordered=[cached[c["id"]] for c in full["controls"] if c["enabled"]]
    return compose_context(trace.base_context,ordered),output.selected,cost
