def augment(api, payload):
    if "LOG_QUERY_RESULT" not in payload["context"]:
        return {"suffix":"","selected":False}
    diagnosis=api.model("Diagnose the next action using only this context:\n"+payload["context"])
    return {"suffix":"\n[Target diagnosis]\n"+diagnosis,"selected":True}
