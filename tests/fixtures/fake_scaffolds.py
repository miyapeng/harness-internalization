"""Injected subprocesses only. Nothing in this module invokes a real coding CLI."""
import json
from types import SimpleNamespace

from fixtures.fake_claude import editing_session
from internalization.evolution.scaffolds.claude_code import ClaudeCodeScaffold
from internalization.evolution.scaffolds.codex import CodexScaffold
from internalization.evolution.scaffolds.qwen_code import QwenCodeScaffold
from internalization.evolution.proposer_profiles import resolve_profile

PROFILE_IDS = {'claude_code': 'claude-sonnet5_claude-code_v1', 'codex': 'gpt-codex_codex_v1',
               'qwen_code': 'qwen-coder_qwen-code_v1'}


def fake_profile(name):
    result = resolve_profile({'profile': PROFILE_IDS[name]})
    result.update(model='fake-provider-model', cli_version='fake-cli-1')
    return result


class FakeCapsule:
    """Explicit test injection, never reachable through a production profile."""
    def preflight(self): pass
    def wrap(self, command, workspace, home): return command


def fake_scaffold(name, rows, calls, *, mutate=None, exit_code=0, timeout=False):
    runner = editing_session(rows, calls, mutate=mutate, exit_code=exit_code, timeout=timeout)
    if name == 'claude_code': return ClaudeCodeScaffold(runner)
    cls = CodexScaffold if name == 'codex' else QwenCodeScaffold
    def factory(argv, **kwargs):
        process = runner.process_factory(argv, **kwargs)
        original = process.communicate
        def communicate(input=None, timeout=None):
            if input is not None: calls[-1]['stdin_text']=input
            stdout, stderr = original(input, timeout)
            events = [json.loads(line) for line in stdout.splitlines()]
            if name == 'codex':
                events = [{'type':'thread.started','thread_id':'fake-session'},
                    {'type':'item.completed','item':{'id':'edit','type':'file_change',
                     'changes':[{'path':'candidate_0/prompts/system.txt','kind':'update'}]}},
                    {'type':'turn.completed','usage':{'input_tokens':110,'output_tokens':20,'cached_input_tokens':7}}]
            else:
                events[0]['subtype'] = 'session_start'
                events[-1].pop('total_cost_usd')
                for block in events[1]['message']['content']:
                    block['name'] = {'Read':'read_file','Write':'write_file'}[block['name']]
            return '\n'.join(json.dumps(row) for row in events), stderr
        process.communicate = communicate
        return process
    return cls(which=lambda _: '/usr/bin/'+cls.executable,
               run_command=lambda argv,**kwargs: SimpleNamespace(stdout='fake-cli-1' if argv[-1]=='--version' else ' '.join(cls.required_flags)),
               process_factory=factory, capsule=FakeCapsule())
