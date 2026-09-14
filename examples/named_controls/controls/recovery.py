def run(api, payload):
    selected = "LOG_QUERY_RESULT" in payload["context"]
    return {"suffix": "\n[Recovery] Use the observed log result; avoid repeating failed actions." if selected else "", "selected": selected}
