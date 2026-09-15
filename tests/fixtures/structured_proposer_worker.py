"""Production JSON entrypoint with an injected fake Claude process, never real Claude."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from internalization.evolution import claude_proposer
from internalization.training.entrypoint import main
from fixtures.fake_claude import editing_session

output = Path(sys.argv[sys.argv.index('--response') + 1]).parent
original = claude_proposer.ClaudeCodeProposer
calls = []

class ScriptedProposer(original):
    def __init__(self, *args, **kwargs):
        rows = json.loads((output/'mock_response.json').read_text())['candidates']
        super().__init__(*args, **kwargs, runner=editing_session(rows,calls))

claude_proposer.ClaudeCodeProposer = ScriptedProposer
try:
    main('propose')
finally:
    (output/'claude_invocations.jsonl').write_text(''.join(json.dumps(c['argv'])+'\n' for c in calls))
