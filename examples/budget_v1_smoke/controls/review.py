def run(api, payload):
    advice=api.model('Check the next action against the task and current public evidence. '
        'Do not invent observations or answers. Give concise internal guidance.\n'+payload['context'])
    return {'suffix':'\n[Public-state review]\n'+advice,'selected':True}
