import importlib
import json


def run(api, payload):
    registry=json.load(open("config/tools.json"))
    if payload["operation"]=="prepare":
        return {"prompt":payload["history"]+"\n"+open("prompts/system.txt").read(),
                "tools":registry,"memory":payload["memory"]}
    action=payload["action"]
    try: item=json.loads(action)
    except ValueError: item={}
    if isinstance(item,dict) and item.get("tool") in registry:
        module,function=registry[item["tool"]]["entry"].split(":")
        handler=getattr(importlib.import_module(module),function)
        result=handler(api,{"history":payload["history"],"arguments":item.get("arguments",{}),"memory":payload["memory"]})
        return {"observation":result,"memory":payload["memory"],"stop":False}
    outcome=api.environment(action)
    return {"observation":outcome["observation"],"memory":payload["memory"],"stop":outcome["done"]}
