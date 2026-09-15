"""Injected process double. No Claude executable or network call is ever used."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

from internalization.evolution.claude_code import ClaudeCodeRunner
from internalization.evolution.claude_tool_guard import ALLOWED_TOOLS

HELP = '--bare --settings --tools --allowedTools --max-turns --setting-sources --disable-slash-commands --strict-mcp-config --plugin-dir'


def editing_session(rows, calls, *, timeout=False, exit_code=0, mutate=None):
    def factory(argv, **kwargs):
        workspace = Path(kwargs['cwd'])
        calls.append({'argv': argv, **{k: v for k, v in kwargs.items() if k not in ('stdin', 'stdout', 'stderr')}})
        initial = {p.relative_to(workspace/'parent').as_posix(): p.read_text() for p in (workspace/'parent').rglob('*') if p.is_file()}
        for name in ('candidate_0', 'candidate_1'):
            assert {p.relative_to(workspace/name).as_posix(): p.read_text() for p in (workspace/name).rglob('*') if p.is_file()} == initial
        metadata, blocks, results = [], [], []
        for i, row in enumerate(rows):
            for edit in row['patch']:
                path = workspace/f'candidate_{i}'/edit['path']
                path.parent.mkdir(parents=True, exist_ok=True)
                if edit['content'] is None:
                    path.unlink()
                    continue  # Deletion is used only by host fixture tests, not exposed as a Claude tool.
                path.write_text(edit['content'])
                tid = f'edit-{len(blocks)}'
                blocks.append({'type': 'tool_use', 'id': tid, 'name': 'Write', 'input': {'file_path': str(path), 'content': edit['content']}})
                results.append({'type': 'tool_result', 'tool_use_id': tid, 'content': 'written', 'is_error': False})
            metadata.append({k: v for k, v in row.items() if k != 'patch'} | {'workspace': f'candidate_{i}'})
        (workspace/'proposal.json').write_text(json.dumps({'candidates': metadata}))
        if mutate: mutate(workspace)
        blocks.append({'type': 'tool_use', 'id': 'read', 'name': 'Read', 'input': {'file_path': str(workspace/'evidence/scores.json')}})
        results.append({'type': 'tool_result', 'tool_use_id': 'read', 'content': '1→public scores', 'is_error': False})
        usage = {'input_tokens': 100, 'output_tokens': 20, 'cache_creation_input_tokens': 3, 'cache_read_input_tokens': 7}
        events = [
            {'type': 'system', 'subtype': 'init', 'session_id': 'fake-session', 'model': argv[argv.index('--model')+1], 'tools': list(ALLOWED_TOOLS)},
            {'type': 'assistant', 'message': {'id': 'message-1', 'usage': usage, 'content': blocks}},
            {'type': 'user', 'message': {'content': results}},
            {'type': 'result', 'subtype': 'success', 'session_id': 'fake-session', 'is_error': False, 'usage': usage, 'total_cost_usd': .03}]
        stdout = '\n'.join(json.dumps(e) for e in events)+'\n'
        class Process:
            returncode = exit_code
            killed = False
            def communicate(self, input=None, timeout=None):
                if timeout is not None and fail_timeout: raise subprocess.TimeoutExpired(argv, timeout)
                return stdout, 'scripted stderr'
            def kill(self): self.killed = True; self.returncode = -9
        fail_timeout = timeout
        process = Process(); calls[-1]['process'] = process
        return process
    return ClaudeCodeRunner(which=lambda _: '/fake/claude',
        run_command=lambda argv, **_: SimpleNamespace(stdout='fake-cli-1' if argv[-1] == '--version' else HELP),
        process_factory=factory)
