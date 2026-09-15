"""hotpotqa H0: use the benchmark adapter's public prompt and action contract.

No extra control/model calls or local JSON tool wrapper. Provenance and the
fixed adapter contract are protected outside this editable workspace.
"""


def run(api, payload):
    if payload["operation"] == "prepare":
        return {"prompt": payload["history"], "tools": {}, "memory": payload["memory"]}
    outcome = api.environment(payload["action"])
    return {"observation": outcome["observation"], "memory": payload["memory"], "stop": outcome["done"]}
