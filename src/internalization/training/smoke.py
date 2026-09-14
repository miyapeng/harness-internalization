"""Independent real two-update smoke, never a retirement or improvement experiment."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import importlib.util
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import time

from ..core.types import Cost, write_json
from ..core.execution_config import resolve_execution, save_effective


def parameter_hash(model):
    hasher=hashlib.sha256()
    for name,parameter in model.named_parameters():
        hasher.update(name.encode())
        hasher.update(parameter.detach().cpu().contiguous().view(-1).view(__import__('torch').uint8).numpy().tobytes())
    return hasher.hexdigest()


def preflight(checkpoint,manifest_path,benchmark,env_config):
    errors=[]
    dependencies={k:importlib.util.find_spec(k) is not None for k in ('torch','transformers','verl')}
    if benchmark=='alfworld': dependencies['alfworld']=importlib.util.find_spec('alfworld') is not None
    for name,available in dependencies.items():
        if not available: errors.append('missing dependency: '+name)
    try: verl_version=version('verl')
    except PackageNotFoundError: verl_version=None
    if verl_version!='0.5.0': errors.append('requires external verl==0.5.0')
    gpu={'available':False,'visible_count':0,'devices':[]}
    if dependencies['torch']:
        import torch
        gpu={'available':torch.cuda.is_available(),'visible_count':torch.cuda.device_count(),'devices':[]}
        if gpu['available']:
            gpu['devices']=[{'index':i,'name':torch.cuda.get_device_name(i),
                'memory_total_bytes':torch.cuda.get_device_properties(i).total_memory} for i in range(torch.cuda.device_count())]
    if not gpu['available']: errors.append('no usable CUDA device; no CPU/mock substitution')
    try:
        from .teacher_backend import checkpoint_fingerprint
        fingerprint=checkpoint_fingerprint(checkpoint)
    except Exception as exc: fingerprint=None; errors.append('checkpoint: '+str(exc))
    try:
        from ..core.manifest import TaskManifest
        manifest=TaskManifest.load(manifest_path)
        if manifest.benchmark!=benchmark or len(manifest.partition('train'))<8: raise ValueError('Need eight actual train tasks')
    except Exception as exc: errors.append('manifest: '+str(exc))
    if benchmark!='alfworld':
        try:
            from ..benchmarks.common import BenchmarkConfig,Catalog
            config=BenchmarkConfig.load(env_config)
            if config.benchmark!=benchmark: raise ValueError('Benchmark mismatch')
            Catalog(config.catalog,benchmark)
            if not Path(config.options['python']).is_file(): raise ValueError('Missing isolated environment Python')
        except Exception as exc: errors.append('environment: '+str(exc))
    elif not env_config.is_file(): errors.append('Missing ALFWorld configuration')
    return {'status':'ready_for_attempt' if not errors else 'not_run','errors':errors,
            'dependencies':dependencies,'verl_version':verl_version,'hardware':gpu,'checkpoint_fingerprint':fingerprint}


def execute(args,execution,report):
    import torch
    from .verl_backend import VerlPolicy,ACTOR_CONFIG
    from .teacher_backend import FrozenHFBackend
    from .behavior_policy import BehaviorPolicySnapshot
    from .trainer import ModuleTrainer
    from .module_advantage import AdvantageConfig,combine_advantages
    from .rollout import InteractionTaskRunner
    from ..core.manifest import TaskManifest
    from ..core.sampling import search_schedule,seed_process
    from ..harness.revision import RevisionStore,InternalizationTarget
    from ..core.trajectory import read_trace_file
    cfg=AdvantageConfig(**execution['advantage'])
    audit=[]
    class AuditedPolicy(VerlPolicy):
        def update(self,batch):
            data=batch.tensor_data
            for tensor in data.values():
                if tensor.is_floating_point() and not torch.isfinite(tensor).all(): raise ValueError('NaN/Inf in actor input')
            expected=combine_advantages(data['token_level_rewards'],data['old_log_probs'],data['module_log_probs'],
                data['response_mask'],data['module_mask'],batch.metadata['task_groups'],cfg)
            if not torch.equal(expected,data['advantages']) or data['advantages'].requires_grad:
                raise ValueError('Actor advantage or gradient isolation mismatch')
            probe_cost=Cost()
            errors=[]
            with BehaviorPolicySnapshot(self) as old:
                for i,prompt in enumerate(batch.metadata['student_prompts']):
                    n=int(data['response_mask'][i].sum())
                    lp,cost=old.score(prompt,data['responses'][i,:n].tolist());probe_cost+=cost
                    error=(torch.tensor(lp)-data['old_log_probs'][i,:n]).abs().max().item()
                    errors.append(error)
            if max(errors)>1e-5: raise ValueError('Cached behavior-policy scoring mismatch')
            before=parameter_hash(self.model);reference_before=parameter_hash(self.reference.model)
            metrics=super().update(batch)
            self.last_update_cost+=probe_cost
            after=parameter_hash(self.model);reference_after=parameter_hash(self.reference.model)
            if before==after: raise ValueError('Actor parameters did not change')
            if reference_before!=reference_after or (audit and reference_before!=audit[0]['reference_before']): raise ValueError('KL reference parameters changed')
            audit.append({'actor_before':before,'actor_after':after,'reference_before':reference_before,
                'reference_after':reference_after,'same_policy_max_error':max(errors),'advantage_reaches_actor':True,
                'selected_steps':int(data['module_mask'].sum()),'module_signal_l1':float(((data['module_log_probs']-
                    data['old_log_probs'])*data['response_mask']*data['module_mask'][:,None]).abs().sum()),
                'optimizer_steps':self.optimizer_steps})
            write_json(args.output/f'update_{len(audit):02d}.json',audit[-1])
            return metrics
    seed_process(execution['seeds']['model_sampling_seed'])
    manifest=TaskManifest.load(args.manifest)
    tasks=search_schedule(manifest.partition('train'),execution['seeds']['run_seed'],cycles=1,count=8)[0]
    write_json(args.output/'smoke_tasks.json',{'source_partition':'train','task_ids':tasks,'manifest_hash':manifest.fingerprint})
    store=RevisionStore(args.output/'revisions')
    full=store.import_directory(args.harness_workspace)
    target=InternalizationTarget.from_control(store,full,args.control_id,'Fixed public-state review for wiring smoke')
    write_json(args.output/'internalization_target.json',target.to_dict())
    if args.benchmark=='alfworld':
        from ..benchmarks.alfworld import AlfworldEnvironment
        factory=lambda:AlfworldEnvironment(args.env_config)
    else:
        from dataclasses import replace
        from ..benchmarks.common import BenchmarkConfig,Catalog
        from ..benchmarks.isolated import environment_factory
        env=BenchmarkConfig.load(args.env_config);catalog=Catalog(env.catalog,args.benchmark)
        env=replace(env,max_steps=execution['max_steps'],**execution['model'],
                    options={**env.options,'catalog_hash':catalog.fingerprint})
        for task in tasks: catalog.get(task,training=True)
        factory=environment_factory(env,args.output,training=True)
    runner=InteractionTaskRunner(factory,max_steps=execution['max_steps'],supervision=execution['supervision'],
                                 model_sampling_seed=execution['seeds']['model_sampling_seed'])
    model_options=execution['model']
    policy=AuditedPolicy(str(args.checkpoint),device=execution['device'],**model_options,**execution['optimizer'])
    reference=FrozenHFBackend(str(args.checkpoint),device=execution['reference_device'],**model_options)
    torch.cuda.reset_peak_memory_stats(policy.model.device)
    trainer=ModuleTrainer(runner,config=cfg,tasks_per_batch=4,rollouts_per_task=4,supervision=execution['supervision'],
        sampling_seed=execution['seeds']['run_seed'],sampling_state={},environment_seed=execution['seeds']['environment_seed'])
    start=time.monotonic()
    checkpoint=trainer.train(policy,reference,full,target.reduced_revision,target=target,tasks=tasks,budget=2,output=args.output/'training')
    if trainer.last_summary['actor_update_calls']!=2 or not trainer.last_summary['optimizer_steps']:
        raise ValueError('Smoke requires two real actor update calls and actual optimizer steps')
    for path in (args.output/'training').glob('rollout_*/trajectories.jsonl'):
        for trajectory in read_trace_file(path):
            if not hasattr(trajectory,'environment_events'): raise ValueError('Complete environment event ledger required')
            # The fixed smoke entrypoint passes through environment context unchanged.
            # Reconstruct by event occurrence; repeated error strings remain separate events.
            context=trajectory.initial_observation
            for step in trajectory.transitions:
                if step.student_prompt!=context: raise ValueError('Unexpected duplicated/modified smoke context')
                events=[e for e in trajectory.environment_events if e.step==step.state.step]
                if len(events)!=1: raise ValueError('Fixed smoke entrypoint must dispatch exactly one action')
                event=events[0]
                if event.observation_kind=='context': context=event.observation
                else: context+='\n[Student action]\n'+step.action+'\n[Environment observation]\n'+event.observation
            if abs(trajectory.total_reward-sum(e.reward for e in trajectory.environment_events))>1e-8:
                raise ValueError('Environment reward missing from training return')
    resource={'used_gpu_count':1,'world_size':1,'device':str(policy.model.device),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(policy.model.device),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(policy.model.device),
        'training_wall_time_s':time.monotonic()-start}
    write_json(args.output/'model_calls.json',{'student':policy.call_stats,'kl_reference':reference.call_stats})
    summary=trainer.last_summary
    # Release actor, optimizer and reference before loading the saved policy.
    del policy,reference,trainer
    gc.collect();torch.cuda.empty_cache()
    reloaded=FrozenHFBackend(checkpoint,device=execution['device'],**model_options)
    runner.sampling_batch=summary["sampling_state"]["draws"]//4
    rollout=runner.rollout(reloaded,target.reduced_revision,tasks[:1],seeds=(execution['seeds']['environment_seed'],),
                           output=args.output/'reloaded_rollout',training=True)
    write_json(args.output/'reload_model_calls.json',reloaded.call_stats)
    return {**report,'status':'passed_real_gpu','training':summary,'updates':audit,'resources':resource,
        'ppo_config':ACTOR_CONFIG,'reloaded_checkpoint':checkpoint,'reload_episodes':len(rollout.trajectories),
        'reload_cost':asdict(sum((t.cost for t in rollout.trajectories),Cost())),
        'context_contract':'fixed smoke prompts exactly reconstructed from event occurrences; overflow raises; no silent token truncation',
        'retirement_or_test_access':False,'performance_or_retirement_claim':False}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--benchmark',choices=('alfworld','webshop','hotpotqa'),required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--env-config',type=Path,required=True)
    p.add_argument('--experiment-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--harness-workspace',type=Path,default=Path('examples/budget_v1_smoke'))
    p.add_argument('--control-id',default='public_review_v1')
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    execution=resolve_execution(json.loads(args.experiment_config.read_text()))
    if execution['mode']!='internalization' or not execution['device'].startswith('cuda'):
        raise ValueError('Real smoke requires internalization on one CUDA device')
    if (execution['tasks_per_batch'],execution['rollouts_per_task'])!=(4,4): raise ValueError('Smoke uses 4 tasks x 4 fresh rollouts')
    save_effective(args.output,execution)
    write_json(args.output/'smoke_protocol.json',{'planned_update_batches':2,'train_pool_size':8,
        'purpose':'independent_wiring_check','formal_schedule_unmodified':execution['schedule']})
    report=preflight(args.checkpoint,args.manifest,args.benchmark,args.env_config)
    write_json(args.output/'preflight.json',report)
    if report['errors'] or args.preflight_only:
        report.update(status='not_run',preflight_only=args.preflight_only)
        write_json(args.output/'smoke_report.json',report)
        print(json.dumps(report,indent=2));return 2 if report['errors'] else 0
    try: report=execute(args,execution,report)
    except Exception as exc:
        report.update(status='failed_real_attempt',error=repr(exc))
        write_json(args.output/'smoke_report.json',report)
        raise
    write_json(args.output/'smoke_report.json',report)
    print(json.dumps(report,indent=2))
    return 0

if __name__=='__main__': raise SystemExit(main())
