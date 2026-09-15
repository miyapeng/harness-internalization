"""Minimal Claude Code process adapter; no candidate or benchmark execution here.

Adapted from Meta-Harness reference_examples/terminal_bench_2/claude_wrapper.py
at 0cbc31e97c9e6d24232d1dc754827c02e1ec415c, MIT.
Copyright (c) 2026 Yoonho Lee. See licenses/Meta-Harness-MIT.txt and
THIRD_PARTY_NOTICES.md. Session invocation, event accounting and tool/file logging
are adapted; workspace policy, strict failures and dependency injection are local.
"""
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import time

from .claude_tool_guard import ALLOWED_TOOLS, check_tool
from ..core.types import Cost, write_json
from .scaffolds.base import ScaffoldPreflightError

TIMEOUT_SECONDS = 600
FORBIDDEN_TOOLS = ('Bash', 'Agent', 'WebSearch', 'WebFetch', 'mcp__*')


class ClaudeCodePreflightError(ScaffoldPreflightError): pass


@dataclass
class ClaudeCodeSessionResult:
    model: str
    cwd: str
    session_id: str = ''
    token_usage: dict = field(default_factory=dict)
    total_cost_usd: float | None = None
    duration_seconds: float = 0.0
    model_calls: int = 0
    files_read: dict = field(default_factory=dict)
    files_written: dict = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)
    raw_events: list = field(default_factory=list)
    stderr: str = ''
    exit_code: int | None = None
    timed_out: bool = False
    errors: list = field(default_factory=list)
    cli_version: str = ''
    command: list = field(default_factory=list)
    allowed_tools: tuple = ALLOWED_TOOLS
    spec_hash: str = ''
    skill_hash: str = ''
    result_received: bool = False

    @property
    def cost(self):
        # Anthropic cache input counters are separate from uncached input tokens.
        inputs = sum(self.token_usage.get(k, 0) for k in
            ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'))
        return Cost(inputs, self.token_usage.get('output_tokens', 0), self.model_calls,
                    self.model_calls, len(self.tool_calls), self.duration_seconds)

    @property
    def succeeded(self):
        return self.exit_code == 0 and self.result_received and not self.timed_out and not self.errors


def parse_stream(stdout, result):
    """Final usage replaces cumulative message usage; never add both totals."""
    messages, calls, final_usage = {}, {}, {}
    initialized = False
    for line in stdout.splitlines():
        if not line.strip(): continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict): raise ValueError('event must be an object')
        except (ValueError, TypeError) as exc:
            result.errors.append('Invalid stream-json: '+str(exc)); continue
        result.raw_events.append(event)
        result.session_id = event.get('session_id', result.session_id)
        kind = event.get('type')
        if kind == 'system' and event.get('subtype') == 'init':
            initialized = True
            if event.get('model') != result.model: result.errors.append('CLI model differs from configured exact ID')
            if set(event.get('tools', [])) != set(ALLOWED_TOOLS): result.errors.append('CLI effective tools differ from fixed allowlist')
            if event.get('mcp_servers') or event.get('plugins'): result.errors.append('Unexpected MCP/plugin configuration')
        if kind == 'assistant':
            message = event.get('message', {})
            messages[message.get('id', f'event-{len(result.raw_events)}')] = message.get('usage', {})
            for block in message.get('content', []):
                if isinstance(block, dict) and block.get('type') == 'tool_use':
                    tool_id = block.get('id', f'tool-{len(calls)}')
                    calls[tool_id] = {'id': tool_id, 'name': block.get('name'),
                        'input': block.get('input', {}), 'output': None, 'is_error': None}
        elif kind == 'user':
            for block in event.get('message', {}).get('content', []):
                if isinstance(block, dict) and block.get('type') == 'tool_result' and block.get('tool_use_id') in calls:
                    call = calls[block['tool_use_id']]
                    call.update(output=block.get('content'), is_error=bool(block.get('is_error', False)))
        elif kind == 'result':
            if result.result_received: result.errors.append('Multiple terminal result events')
            result.result_received = True
            result.total_cost_usd = event.get('total_cost_usd')
            final_usage = event.get('usage', {})
            if event.get('is_error') or event.get('subtype') != 'success':
                result.errors.append('CLI result failed: '+str(event.get('subtype')))
    result.model_calls = len(messages)
    keys = ('input_tokens', 'output_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens')
    for key in keys:
        if key in final_usage or any(key in usage for usage in messages.values()):
            result.token_usage[key] = final_usage.get(key, sum(usage.get(key, 0) for usage in messages.values()))
    for key, value in list(result.token_usage.items()):
        if type(value) is not int or value < 0:
            result.errors.append('Invalid token counter: '+key); result.token_usage[key] = 0
    if result.total_cost_usd is not None and (type(result.total_cost_usd) not in (int, float) or not math.isfinite(result.total_cost_usd) or result.total_cost_usd < 0):
        result.errors.append('Invalid CLI cost counter'); result.total_cost_usd = None
    result.tool_calls = list(calls.values())
    for call in result.tool_calls:
        try: relative = check_tool(result.cwd, call['name'], call['input'])
        except ValueError as exc:
            if call['is_error'] is not True: result.errors.append('Unblocked tool boundary violation: '+str(exc))
            continue
        if call['name'] in ('Read', 'Write', 'Edit'):
            stats = result.files_read if call['name'] == 'Read' else result.files_written
            row = stats.setdefault(relative, {'attempts': 0, 'completed': 0})
            row['attempts'] += 1
            row['completed'] += int(call['is_error'] is False)
    if not result.result_received: result.errors.append('No terminal CLI result')
    if not initialized: result.errors.append('No CLI initialization/permission evidence')
    if not result.session_id: result.errors.append('No CLI session ID')
    return result


class ClaudeCodeRunner:
    def __init__(self, *, timeout=TIMEOUT_SECONDS, which=shutil.which,
                 run_command=subprocess.run, process_factory=subprocess.Popen):
        if timeout <= 0: raise ValueError('Positive Claude timeout required')
        self.timeout, self.which = timeout, which
        self.run_command, self.process_factory = run_command, process_factory
        self.last_result = None

    def preflight(self):
        binary = self.which('claude')
        if not binary: raise ClaudeCodePreflightError('Claude Code preflight: claude executable not found on PATH')
        try:
            version = self.run_command([binary, '--version'], capture_output=True, text=True, timeout=10, check=True)
            help_result = self.run_command([binary, '--help'], capture_output=True, text=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ClaudeCodePreflightError('Claude Code preflight: CLI version/help unavailable') from exc
        required = ('--bare', '--settings', '--tools', '--allowedTools', '--max-turns',
                    '--setting-sources', '--disable-slash-commands', '--strict-mcp-config', '--plugin-dir')
        if any(flag not in help_result.stdout for flag in required):
            raise ClaudeCodePreflightError('Claude Code preflight: required isolation/limit flags unavailable; no fallback')
        return binary, version.stdout.strip()

    def run(self, *, model, max_turns, cwd, output, spec, skill, preflight=None, provider=None, max_tool_calls=None):
        cwd, output = Path(cwd).resolve(strict=True), Path(output).resolve()
        if cwd.name not in ('proposer_workspace', 'claude_workspace'): raise ValueError('Claude cwd must be a dedicated proposer workspace')
        if type(max_turns) is not int or max_turns <= 0: raise ValueError('Positive max_turns required')
        binary, version = preflight or self.preflight()
        output.mkdir(parents=True, exist_ok=True)
        empty = output/'empty_plugins'; empty.mkdir()
        config = output/'isolated_config'; config.mkdir()
        guard = Path(__file__).with_name('claude_tool_guard.py').resolve()
        hook_command = shlex.join([sys.executable, '-I', '-S', str(guard), str(cwd), str(output/'tool_permissions.jsonl')] + ([str(max_tool_calls)] if max_tool_calls is not None else []))
        settings = {'autoMemoryEnabled': False, 'claudeMdExcludes': ['/**'],
            'hooks': {'PreToolUse': [{'matcher': '*', 'hooks': [{'type': 'command', 'command': hook_command}]}]}}
        settings_path = output/'settings.json'; write_json(settings_path, settings)
        command = [binary, '-p', '--bare', '--output-format', 'stream-json', '--verbose',
            '--model', model, '--max-turns', str(max_turns), '--setting-sources', '',
            '--tools', ','.join(ALLOWED_TOOLS), '--allowedTools', ','.join(ALLOWED_TOOLS),
            '--disallowedTools', ','.join(FORBIDDEN_TOOLS), '--permission-mode', 'dontAsk',
            '--disable-slash-commands', '--strict-mcp-config', '--mcp-config', '{"mcpServers":{}}',
            '--plugin-dir', str(empty), '--settings', str(settings_path),
            '--append-system-prompt', spec+'\n\nEXPLICIT WORKFLOW\n'+skill]
        # No ambient injection, alternate model endpoints, or task-model credentials.
        allowed_env = ('PATH', 'LANG', 'LC_ALL', 'ANTHROPIC_API_KEY', 'HTTP_PROXY', 'HTTPS_PROXY',
                       'ALL_PROXY', 'NO_PROXY', 'http_proxy', 'https_proxy', 'all_proxy', 'no_proxy',
                       'SSL_CERT_FILE', 'SSL_CERT_DIR', 'NODE_EXTRA_CA_CERTS')
        env = {k: os.environ[k] for k in allowed_env if k in os.environ}
        if provider is not None:
            # Only the explicitly resolved provider may set an endpoint or credential.
            env.pop('ANTHROPIC_API_KEY', None)
            key = os.environ.get(provider['api_key_env'])
            if key: env['ANTHROPIC_API_KEY'] = key
            env['ANTHROPIC_BASE_URL'] = provider['base_url']
            # Disable model aliases/fallback routing to unrelated providers.
            env['ANTHROPIC_DEFAULT_SONNET_MODEL'] = model
            env['ANTHROPIC_DEFAULT_HAIKU_MODEL'] = model
            env['ANTHROPIC_DEFAULT_OPUS_MODEL'] = model
        env.update(CLAUDE_CONFIG_DIR=str(config), CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1')
        result = ClaudeCodeSessionResult(model=model, cwd=str(cwd), cli_version=version, command=command,
            spec_hash=hashlib.sha256(spec.encode()).hexdigest(), skill_hash=hashlib.sha256(skill.encode()).hexdigest())
        started = time.monotonic(); stdout = ''; stderr = ''
        try:
            process = self.process_factory(command, cwd=str(cwd), env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate('Read PROPOSER_SPEC.md and follow the injected workflow. Write proposal.json, then stop.\n', timeout=self.timeout)
            except subprocess.TimeoutExpired:
                result.timed_out = True
                if isinstance(process, subprocess.Popen):
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                else: process.kill()  # Injected test process; never send a host PID signal.
                stdout, stderr = process.communicate()
            result.exit_code = process.returncode
        except OSError as exc:
            result.errors.append('CLI execution error: '+str(exc))
        finally:
            result.duration_seconds = time.monotonic()-started
            result.stderr = stderr or ''
            stdout = stdout or ''
            try: parse_stream(stdout, result)
            except (ValueError, TypeError, AttributeError, KeyError) as exc:
                result.errors.append('Malformed CLI event data: '+str(exc))
                result.token_usage = {}; result.total_cost_usd = None
            self.last_result = result
            (output/'stdout.jsonl').write_text(stdout)
            (output/'stderr.log').write_text(result.stderr)
            write_json(output/'session.json', asdict(result))
            write_json(output/'cost.json', asdict(result.cost))
        return result
