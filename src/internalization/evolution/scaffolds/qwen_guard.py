"""Qwen native tool argument adaptation to the shared trusted path guard.

Loaded explicitly by host settings in an isolated home, never a discovered skill.
"""
import json
from pathlib import Path
import sys

# The host copies the unchanged shared file guard alongside this trusted script.
sys.path.insert(0, str(Path(__file__).parent))
from file_guard import check_tool

TOOLS = {'read_file': 'Read', 'write_file': 'Write', 'edit': 'Edit',
         'glob': 'Glob', 'grep_search': 'Grep', 'search_file_content': 'Grep', 'list_directory': 'Glob'}


def main():
    workspace, audit = sys.argv[1:]
    try:
        event = json.load(sys.stdin); name = event['tool_name']; arguments = dict(event.get('tool_input', {}))
        if name not in TOOLS: raise ValueError('Qwen tool outside fixed native file-tool allowlist')
        if name in ('read_file', 'write_file', 'edit'):
            arguments['file_path'] = arguments.get('file_path', arguments.get('absolute_path', arguments.get('path')))
        path = check_tool(workspace, TOOLS[name], arguments)
        decision, reason = 'allow', 'Scoped native file operation'
    except Exception as exc:
        decision, reason, path = 'deny', str(exc), None
    with Path(audit).open('a') as stream:
        stream.write(json.dumps({'decision': decision, 'path': path, 'reason': reason})+'\n')
    print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse',
        'permissionDecision': decision, 'permissionDecisionReason': reason}}))


if __name__ == '__main__': main()
