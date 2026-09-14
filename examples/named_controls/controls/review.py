def run(api, payload):
    if "LOG_QUERY_RESULT" not in payload["context"]:
        return {"suffix": "", "selected": False}
    guidance = api.model("Review the next action using only this public context:\n" + payload["context"])
    return {"suffix": "\n[Review] " + guidance, "selected": True}
