"""Exercise the production JSON entrypoint with a scripted, counted API transport."""
import json
from pathlib import Path
import sys

from internalization.evolution import code_proposer
from internalization.training.entrypoint import main

output = Path(sys.argv[sys.argv.index('--response') + 1]).parent
original = code_proposer.CodeProposer


def transport(payload):
    with (output / 'transport_calls.jsonl').open('a') as stream:
        stream.write(json.dumps(payload) + '\n')
    return {'choices': [{'message': {'content': (output / 'mock_response.json').read_text()}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 20}}


class ScriptedProposer(original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, transport=transport)


code_proposer.CodeProposer = ScriptedProposer
main('propose')
