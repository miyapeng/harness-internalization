"""Official Codex exec adapter. No OpenAI agent loop or upstream code is vendored."""
import json
from .cli_process import NativeCLI
from .base import ScaffoldPreflightError


class CodexScaffold(NativeCLI):
    scaffold_id = 'codex'
    executable = 'codex'
    credential_env = 'OPENAI_API_KEY'
    help_args = ('exec', '--help')
    required_flags = ('--json', '--sandbox', '--skip-git-repo-check', '--ephemeral', '--ignore-rules', '--ignore-user-config')

    def validate_limits(self, limits):
        # exec JSON has one user turn, not a count of all internal model requests.
        # Never present that value as an equivalent Claude/Qwen max-turns cap.
        if limits['max_turns'] is not None or limits['max_tool_calls'] is not None:
            raise ScaffoldPreflightError('Codex exec cannot enforce these model-turn/tool caps; use explicit null with wall-time or a separately verified runtime')

    def command(self, binary, workspace, home, model, spec, workflow, limits, provider):
        config = home/'.codex'; config.mkdir()
        # All overrides are host-controlled. CLI ignores ambient user config and
        # rules; capsule hides ancestor AGENTS.md and all other experiment files.
        settings = {
            'approval_policy': 'never', 'sandbox_workspace_write.network_access': False,
            'sandbox_workspace_write.exclude_tmpdir_env_var': True,
            'sandbox_workspace_write.exclude_slash_tmp': True,
            'web_search': 'disabled', 'project_doc_max_bytes': 0,
            'features.multi_agent': False, 'features.apps': False,
            'features.skills': False, 'features.js_repl': False,
            'features.exec_policy': False, 'shell_environment_policy.inherit': 'none',
            'model_provider': 'proposer', 'model_providers.proposer.name': provider['id'],
            'model_providers.proposer.base_url': provider['base_url'],
            'model_providers.proposer.wire_api': 'responses',
            'model_providers.proposer.env_key': 'OPENAI_API_KEY',
        }
        command = [binary, 'exec', '--json', '--ephemeral', '--ignore-rules', '--ignore-user-config',
                   '--sandbox', 'workspace-write', '--skip-git-repo-check', '--cd', str(workspace), '--model', model]
        for key, value in settings.items():
            command += ['-c', key+'='+json.dumps(value)]
        command += ['-']
        return command, {'CODEX_HOME': '/home/proposer/.codex'}

    def parse_events(self, result):
        terminal = 0; items = {}
        for event in result.raw_events:
            kind = event.get('type')
            if kind == 'thread.started': result.session_id = event['thread_id']
            elif kind == 'turn.completed':
                terminal += 1; result.completed = True
                result.token_usage = dict(event.get('usage', {}))
                result.total_cost_usd = event.get('total_cost_usd')
            elif kind in ('error', 'turn.failed'): result.errors.append('Codex '+kind)
            elif kind in ('item.started', 'item.updated', 'item.completed'):
                item = event['item']; items[item['id']] = item
        if terminal != 1: result.errors.append('Expected exactly one Codex proposal turn completion')
        result.tool_calls = [item for item in items.values() if item.get('type') in
                            ('command_execution', 'file_change', 'mcp_tool_call', 'web_search', 'collab_tool_call')]
        if any(item.get('type') in ('mcp_tool_call', 'web_search', 'collab_tool_call') for item in result.tool_calls):
            result.errors.append('Forbidden Codex tool observed')
        # Native command events do not expose a reliable complete file-read list.
        result.files_read = None
        result.files_written = {change['path']: {'reported': True}
            for item in result.tool_calls if item.get('type') == 'file_change' for change in item.get('changes', [])}
