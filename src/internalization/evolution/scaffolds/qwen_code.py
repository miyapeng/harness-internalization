"""Official Qwen headless CLI, native file tools, and explicit protected settings."""
import json
from pathlib import Path
import shlex

from .cli_process import NativeCLI
from .base import ScaffoldPreflightError

TOOLS = ('read_file', 'write_file', 'edit', 'glob', 'grep_search', 'search_file_content', 'list_directory')
FORBIDDEN = ('run_shell_command', 'shell', 'agent', 'list_agents', 'send_message', 'task',
             'web_search', 'web_fetch', 'google_web_search', 'skill', 'tool_search', 'save_memory')


class QwenCodeScaffold(NativeCLI):
    scaffold_id = 'qwen_code'
    executable = 'qwen'
    credential_env = 'OPENAI_API_KEY'
    help_args = ('--help',)
    required_flags = ('--prompt', '--output-format', '--max-session-turns', '--max-wall-time',
                      '--max-tool-calls', '--core-tools', '--exclude-tools', '--append-system-prompt')

    def input_text(self, spec, workflow):
        # -p supplies the user instruction; spec/workflow are already appended to
        # the system prompt. Do not inject a duplicate copy through stdin.
        return ''

    def validate_limits(self, limits):
        if limits['max_turns'] is None: raise ScaffoldPreflightError('Qwen requires explicit max-session-turns')

    def command(self, binary, workspace, home, model, spec, workflow, limits, provider):
        config = home/'.qwen'; config.mkdir()
        (home/'guard.py').write_text(Path(__file__).with_name('qwen_guard.py').read_text())
        (home/'file_guard.py').write_text(Path(__file__).resolve().parent.parent.joinpath('claude_tool_guard.py').read_text())
        hook = shlex.join(['/usr/bin/python3', '-I', '-S', '/home/proposer/guard.py',
                          str(workspace), '/home/proposer/tool_permissions.jsonl'])
        settings = {
            'tools': {'core': list(TOOLS), 'disabled': list(FORBIDDEN), 'approvalMode': 'auto-edit'},
            'mcpServers': {}, 'telemetry': {'enabled': False},
            'context': {'fileName': '__host_has_no_ambient_instructions__'},
            'memory': {'enableManagedAutoMemory': False, 'enableAutoSkill': False, 'enableManagedAutoDream': False},
            'hooks': {'PreToolUse': [{'matcher': '*', 'hooks': [{'type': 'command', 'command': hook}]}]},
        }
        (config/'settings.json').write_text(json.dumps(settings))
        command = [binary, '-p', 'Read PROPOSER_SPEC.md and the explicitly appended workflow; write proposal.json and stop.',
            '--model', model, '--output-format', 'stream-json', '--approval-mode', 'auto-edit',
            '--max-session-turns', str(limits['max_turns']), '--max-wall-time', str(limits['max_wall_time_s']),
            '--core-tools', ','.join(TOOLS), '--exclude-tools', ','.join(FORBIDDEN),
            '--append-system-prompt', spec+'\n\nEXPLICIT WORKFLOW\n'+workflow]
        if limits['max_tool_calls'] is not None: command += ['--max-tool-calls', str(limits['max_tool_calls'])]
        return command, {'OPENAI_BASE_URL': provider['base_url'], 'OPENAI_MODEL': model}

    def parse_events(self, result):
        messages, calls = {}, {}; terminal = 0; initialized = False
        for event in result.raw_events:
            result.session_id = event.get('session_id', result.session_id)
            kind = event.get('type')
            if kind == 'system' and event.get('subtype') in ('init', 'session_start'):
                initialized = True
                if event.get('model') != result.model: result.errors.append('Qwen runtime model mismatch')
                if event.get('mcp_servers'): result.errors.append('Unexpected Qwen MCP configuration')
            elif kind == 'assistant':
                message = event['message']; messages[message['id']] = message.get('usage', {})
                for block in message.get('content', []):
                    if block.get('type') == 'tool_use':
                        calls[block['id']] = {'id': block['id'], 'name': block['name'], 'input': block.get('input', {}),
                                               'output': None, 'is_error': None}
            elif kind == 'user':
                for block in event.get('message', {}).get('content', []):
                    if block.get('type') == 'tool_result' and block.get('tool_use_id') in calls:
                        calls[block['tool_use_id']].update(output=block.get('content'), is_error=bool(block.get('is_error', False)))
            elif kind == 'result':
                terminal += 1
                result.completed = event.get('subtype') == 'success' and not event.get('is_error', False)
                result.token_usage = dict(event.get('usage', {})); result.total_cost_usd = event.get('total_cost_usd')
        if not initialized or terminal != 1: result.errors.append('Missing/duplicate Qwen initialization or result')
        result.model_calls = len(messages); result.tool_calls = list(calls.values())
        if not result.token_usage:
            for key in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'):
                if any(key in row for row in messages.values()): result.token_usage[key] = sum(row.get(key, 0) for row in messages.values())
        if any(call['name'] not in TOOLS and call['is_error'] is not True for call in calls.values()):
            result.errors.append('Unexpected unblocked Qwen tool')
        result.files_read = {}; result.files_written = {}
        for call in calls.values():
            if call['name'] not in ('read_file', 'write_file', 'edit'): continue
            args = call['input']; path = args.get('file_path', args.get('absolute_path', args.get('path')))
            if isinstance(path, str):
                stats = result.files_read if call['name'] == 'read_file' else result.files_written
                stats[path] = {'reported': True, 'completed': call['is_error'] is False}
