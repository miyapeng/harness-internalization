def run(api, payload):
    if payload['operation']=='prepare':
        return {'prompt':payload['history'],'tools':{},'memory':payload['memory']}
    outcome=api.environment(payload['action'])
    return {'observation':outcome['observation'],'stop':outcome['done'],'memory':payload['memory']}
