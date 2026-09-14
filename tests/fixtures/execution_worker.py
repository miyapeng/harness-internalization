"""Real JSON subprocess transport with CPU mock policy/environment (no HF/veRL)."""
import json
from pathlib import Path
import sys
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_behavior_policy import BehaviorPolicyTests
from internalization.training import entrypoint, verl_backend
from internalization.training.rollout import EnvironmentStep

constructors = {}
policies = []
request_path = Path(sys.argv[sys.argv.index('--request')+1])
request = json.loads(request_path.read_text())
output = Path(sys.argv[sys.argv.index('--response')+1]).parent
class Environment:
    def __init__(self,*args): self.steps=0
    def reset(self,task,seed): self.steps=0;return 'PUBLIC'
    def step(self,action):
        self.steps+=1
        return EnvironmentStep('repeat error',float(self.steps),False,0.)
    def close(self): pass

def policy_loader(path,**kwargs):
    constructors['policy']=kwargs
    policy=BehaviorPolicyTests().policy()
    policies.append(policy)
    policy.optimizer.param_groups[0]['lr']=kwargs['learning_rate']
    policy.optimizer_steps=0
    def count(*args): policy.optimizer_steps+=1
    policy.optimizer.register_step_post_hook(count)
    return policy

def reference_loader(path,**kwargs):
    constructors['reference']=kwargs
    return BehaviorPolicyTests().reference()

with patch.object(entrypoint,'AlfworldEnvironment',Environment), patch.object(entrypoint,'FrozenHFBackend',reference_loader), patch.object(verl_backend,'VerlPolicy',policy_loader):
    entrypoint.main('train')
(output/'constructors.json').write_text(json.dumps(constructors))

(output/'batch_evidence.json').write_text(json.dumps([{
    'max_abs_advantage':float(b['advantages'].abs().max()),
    'selected':b['module_mask'].tolist(),
    'old_log_probs':b['old_log_probs'].tolist(),
    'module_log_probs':b['module_log_probs'].tolist(),
} for policy in policies for b in policy.batches]))
