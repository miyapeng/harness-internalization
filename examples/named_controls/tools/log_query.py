def run(api, payload):
    needle=payload["arguments"].get("needle","")
    return "LOG_QUERY_RESULT\n"+"\n".join(line for line in payload["history"].splitlines()
        if line.startswith("log:") and needle in line)
