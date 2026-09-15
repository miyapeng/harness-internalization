"""Small executable revision pair for interface tests; no legacy module runtime."""
import json
from internalization.harness.revision import RevisionStore, InternalizationTarget

AGENT = '''def run(api,payload):
    if payload["operation"] == "prepare":
        return {"prompt":payload["history"],"tools":{},"memory":payload["memory"]}
    outcome=api.environment(payload["action"])
    return {"observation":outcome["observation"],"memory":payload["memory"],"stop":outcome["done"]}
'''

def pair(root, *, active=True, condition=None):
    condition = str(active) if condition is None else condition
    hook = ('def run(api,payload):\n'
            f'    if not ({condition}): return {{"suffix":"","selected":False}}\n'
            '    return {"suffix":"\\n\\n[Internal guidance]\\n"+api.model(payload["context"]+"\\n\\nanalyze"),"selected":True}\n')
    files={'agent/main.py':AGENT,'controls/target.py':hook,
           'config/harness.json':json.dumps({'schema':1,'entrypoint':'agent/main.py:run','supervision':'controls/target.py:run'})}
    store=RevisionStore(root)
    full=store.snapshot(files)
    files['config/harness.json']=json.dumps({**full.config,'supervision':None})
    reduced=store.snapshot(files)
    target=InternalizationTarget(full,reduced,'One independently added internal computation','controls/target.py:run')
    target.validate_structure()
    return full,target
