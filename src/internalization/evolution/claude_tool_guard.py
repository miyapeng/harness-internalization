"""Trusted PreToolUse boundary for the five proposer file tools, not a code sandbox.

Invoked by the host's explicit CLI settings, never discovered from candidate files.
The ordinary candidate execution sandbox remains a separate, unchanged component.
"""
import json
from pathlib import Path
import sys

ALLOWED_TOOLS = ('Read', 'Glob', 'Grep', 'Edit', 'Write')


def check_tool(workspace, name, arguments):
    root = Path(workspace).resolve(strict=True)
    if name not in ALLOWED_TOOLS or not isinstance(arguments, dict):
        raise ValueError('Tool outside fixed proposer allowlist')
    value = arguments.get('file_path') if name in ('Read', 'Edit', 'Write') else arguments.get('path', '.')
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('A concrete workspace path is required')
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if '..' in path.parts or any(part.startswith('~') for part in path.parts):
        raise ValueError('Traversal paths are forbidden')
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink(): raise ValueError('Symlink access is forbidden')
    path.resolve().relative_to(root)
    if name in ('Glob', 'Grep'):
        # Glob selectors are paths, whereas Grep.pattern is a regular expression.
        selector = arguments.get('pattern', '') if name == 'Glob' else arguments.get('glob', '')
        # Do not allow brace expansion/escaping to hide an absolute or ../ branch.
        if (not isinstance(selector, str) or selector.startswith('/') or
                any(token in selector for token in ('..', '{', '}', '\\', '~', '\x00'))):
            raise ValueError('Glob selector must stay inside workspace')
    if name in ('Edit', 'Write'):
        if relative.as_posix() != 'proposal.json' and (not relative.parts or relative.parts[0] not in ('candidate_0', 'candidate_1')):
            raise ValueError('Writes are limited to two candidates and proposal.json')
    return relative.as_posix()


def main():
    workspace, audit_path = sys.argv[1:]
    try:
        request = json.load(sys.stdin)
        path = check_tool(workspace, request['tool_name'], request.get('tool_input', {}))
        decision, reason = 'allow', 'Scoped proposer file operation'
    except Exception as exc:
        decision, reason, path = 'deny', str(exc), None
    # Logging failure blocks the operation; no silent permission fallback.
    with Path(audit_path).open('a') as stream:
        stream.write(json.dumps({'decision': decision, 'reason': reason, 'path': path}) + '\n')
    print(json.dumps({'hookSpecificOutput': {'hookEventName': 'PreToolUse',
        'permissionDecision': decision, 'permissionDecisionReason': reason}}))


if __name__ == '__main__': main()
