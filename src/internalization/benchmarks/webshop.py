"""WebShop text environment via its external official simulator and reward.

Original integration code, no bundled upstream engine. Use only inside the isolated
benchmark worker. Source identity is pinned before exposing any public observation.
"""
from dataclasses import asdict
from pathlib import Path
import json
import re
import sys

from .common import Catalog, checked_repo, sha256, tree_hash, score01
from ..core.types import digest, write_json
from ..core.sampling import seed_process
from ..training.rollout import EnvironmentStep

STATUS = 'not_ready_real_official_assets_and_gpu_not_verified'
ASSET_FIELDS={'repo','commit','products','products_sha256','attributes','attributes_sha256',
              'human_attributes','human_attributes_sha256','index_hash','initialization_seed'}


def official_environment(options):
    if set(options)-ASSET_FIELDS-{'python','catalog_hash'} or not ASSET_FIELDS<=set(options):
        raise ValueError('WebShop requires explicit pinned full assets; unknown options forbidden')
    root=checked_repo(options['repo'],options['commit'])
    for name in ('products','attributes','human_attributes'):
        if sha256(options[name])!=options[name+'_sha256']: raise ValueError(f'WebShop {name} identity changed')
    index=root/'search_engine/indexes'
    if tree_hash(index)!=options['index_hash']: raise ValueError('Full WebShop search index identity changed')
    products=json.loads(Path(options['products']).read_text())
    if len(products)<1_000_000: raise ValueError('Full million-product configuration required; debug subsets forbidden')
    del products
    sys.path.insert(0,str(root))
    from web_agent_site.engine import engine
    from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv
    if not Path(engine.__file__).resolve().is_relative_to(root): raise ValueError('Wrong installed WebShop engine')
    # Official loader takes the product file and module-level attribute locations.
    # Configure data paths in this isolated process; do not patch official source or scoring.
    engine.DEFAULT_ATTR_PATH=str(Path(options['attributes']).resolve())
    engine.HUMAN_ATTR_PATH=str(Path(options['human_attributes']).resolve())
    seed_process(options['initialization_seed'])
    return WebAgentTextEnv(observation_mode='text',file_path=str(Path(options['products']).resolve()),
        num_products=None,human_goals=True,filter_goals=None,limit_goals=-1,show_attrs=False,
        get_image=False,num_prev_obs=0,num_prev_actions=0)


def export_tasks(options):
    env=official_environment(options)
    try:
        identity=digest({k:options[k] for k in sorted(ASSET_FIELDS)})
        rows=[]
        for index,goal in enumerate(env.server.goals):
            source='official_test' if index<500 else 'official_eval' if index<1500 else 'official_train'
            rows.append({'id':task_id(index),'split':{'official_test':'test','official_eval':'dev','official_train':'train'}[source],
                'source':source,'source_hash':identity,'goal_hash':digest(goal),
                'group_id':digest(['webshop',goal['asin']]),
                'task_type':str(goal['product_category']), 'session_id':index})
        if len(rows)<=1500: raise ValueError('Incomplete official goal population')
        return rows
    finally: env.close()


class WebShopEnvironment:
    def __init__(self,config,output,*,training=False,environment_loader=official_environment):
        self.config,self.output,self.training=config,Path(output),training
        self.catalog=Catalog(config.catalog,'webshop')
        self.environment_loader=environment_loader
        self.env=None
        self.done=True

    def reset(self,task_id,seed):
        row=self.catalog.get(task_id,self.training)
        if f"webshop:{row['session_id']}"!=task_id: raise ValueError('Explicit official task ID required')
        index=row['session_id']
        expected='official_test' if index<500 else 'official_eval' if index<1500 else 'official_train'
        if index<0 or row['source']!=expected: raise ValueError('WebShop official task split mismatch')
        if self.env is not None: self.env.close()
        self.env=self.environment_loader(self.config.options)
        if digest(self.env.server.goals[index])!=row['goal_hash']: raise ValueError('Goal order or content changed')
        seed_process(seed)
        observation,_=self.env.reset(session=index)  # integer selects exact official goal, never random reset
        self.task_id,self.seed,self.steps,self.done=task_id,seed,0,False
        self.history=['Use search[keywords] or click[value] from the current page.',str(observation)]
        write_json(self.output/'identity.json',{'task_id':task_id,'seed':seed,'catalog_hash':self.catalog.fingerprint,
            'source_hash':row['source_hash'],'num_products':None,'human_goals':True})
        return {'observation':'\n'.join(self.history)}

    def step(self,action):
        if self.done: raise RuntimeError('Episode already finished')
        legal=self.env.get_available_actions()
        match=re.fullmatch(r'(search|click)\[(.+)\]',action.strip(),re.DOTALL)
        valid=bool(match and (match[1]=='search' or match[2].lower() in legal['clickables'] and match[2].lower()!='search'))
        observation,reward,terminal,_=self.env.step(action.strip())
        reward=score01(reward)
        self.steps+=1
        self.done=bool(terminal or self.steps>=self.config.max_steps)
        # Official terminal HTML may contain goal/attribute debug information. It is not an agent observation.
        public='Purchase submitted.' if terminal else str(observation)
        self.history.extend(['Action: '+action,'Observation: '+public])
        if self.done:
            write_json(self.output/'grade.json',{'task_id':self.task_id,'seed':self.seed,
                'metrics':{'reward':reward,'success':float(reward==1)},'reward_metric':'official_webshop_reward',
                'termination':'purchase' if terminal else 'step_limit'})
        return asdict(EnvironmentStep('\n'.join(self.history),reward,self.done,reward,valid,1,observation_kind='context'))

    def close(self):
        if self.env is not None: self.env.close()


# Preserved public task partition helpers.
def task_id(session):
    if type(session) is not int or session < 0: raise ValueError("Invalid WebShop session")
    return f"webshop:{session}"


def split_for(session):
    task_id(session)
    return "test" if session < 500 else "dev" if session < 1500 else "train"


def check_partition(sessions, partition):
    if any(split_for(session) != partition for session in sessions):
        raise ValueError("WebShop task is outside its official partition")
