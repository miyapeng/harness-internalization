"""Revision execution extension of InteractionTaskRunner; same trajectory/token contracts."""
from dataclasses import asdict
import time

from ..harness.code_runtime import CodeRuntime
from ..core.types import Cost, Journal, State
from ..core.trajectory import Trajectory, Transition, RolloutResult


def rollout_revision(runner,model,harness,tasks,*,seeds,output,training=False):
    if isinstance(model,str):
        if runner.model_loader is None: raise ValueError("Model loader required")
        model=runner.model_loader(model)
    harness.files()
    journal=Journal(output/"trajectories.jsonl")
    audit=Journal(output/"revision_execution.jsonl")
    trajectories=[]
    for task in tasks:
        for replica,seed in enumerate(seeds):
            env=runner.environment_factory()
            episode=f"{task}:{seed}"+(f":replica:{replica}" if training else "")
            runtime=CodeRuntime(model,harness,env,sandbox=runner.code_sandbox,audit=audit)
            steps,cost,success=[],Cost(),0.
            started=time.perf_counter()
            try:
                history=env.reset(task,seed)
                for index in range(runner.max_steps):
                    prompt,done=runtime.prepare(history,index)
                    if done:
                        cost+=runtime.broker.cost
                        success=runtime.broker.success
                        break
                    # Only the actually visible context enters same-state teacher scoring.
                    state=State(task,episode,index,prompt)
                    action=model.generate(prompt,purpose="rollout_action" if training else "action")
                    ids=tuple(action.response_ids)
                    old,scoring=(),Cost()
                    if training:
                        if not ids: raise ValueError("Training requires exact student response tokens")
                        old,scoring=model.score(prompt,list(ids))
                    # Persist consumed policy calls before candidate dispatch can fail.
                    audit.append("student_generation",task_id=task,episode_id=episode,step=index,
                        model_version=model.snapshot_id,harness_version=harness.version,
                        cost=asdict(action.cost+scoring))
                    outcome=runtime.execute(action.text,history,index)
                    step_cost=runtime.broker.cost+action.cost+scoring
                    cost+=step_cost
                    transition=Transition(state,prompt,action.text,ids,outcome.reward,outcome.done,
                        outcome.action_valid,tuple(old),step_cost,tuple(model.prompt_ids(prompt)) if training else ())
                    steps.append(transition)
                    history,success=outcome.observation,outcome.success
                    journal.append("transition",state=asdict(state),action=action.text,student_prompt=prompt,
                        response_ids=ids,reward=outcome.reward,done=outcome.done,cost=asdict(step_cost),
                        model_version=model.snapshot_id,harness_version=harness.version,harness_path=harness.path)
                    if outcome.done: break
            except Exception as exc:
                audit.append("candidate_runtime_failure",task_id=task,episode_id=episode,
                    model_version=model.snapshot_id,harness_version=harness.version,
                    elapsed_s=time.perf_counter()-started,error=str(exc))
                raise
            finally: env.close()
            fields=asdict(cost);fields["latency_s"]=time.perf_counter()-started
            trajectory=Trajectory(task,episode,seed,model.snapshot_id,harness.version,tuple(steps),success,Cost(**fields),task)
            trajectories.append(trajectory)
            journal.append("trajectory",trajectory=asdict(trajectory),harness_path=harness.path)
    return RolloutResult(tuple(trajectories),tuple(t.outcome for t in trajectories))
