"""Adapter over the MIT-attributed Claude runner; no duplicate proposal logic."""
from ..claude_code import ClaudeCodeRunner
from .base import ProposerSessionResult, text_hash


class ClaudeCodeScaffold:
    scaffold_id = 'claude_code'

    def __init__(self, runner=None):
        self.runner = runner if runner is not None else ClaudeCodeRunner()
        self._preflight = None

    def preflight(self):
        binary, version = self.runner.preflight()
        self._preflight = (binary, version)
        return {'binary': binary, 'version': version, 'scaffold_id': self.scaffold_id}

    def run(self, *, workspace, model, spec, workflow, limits, output, provider):
        self.runner.timeout = limits['max_wall_time_s']
        native = self.runner.run(model=model, max_turns=limits['max_turns'], cwd=workspace,
            output=output, spec=spec, skill=workflow, preflight=self._preflight,
            provider=provider, max_tool_calls=limits['max_tool_calls'])
        return ProposerSessionResult(self.scaffold_id, native.cli_version, provider, model,
            session_id=native.session_id, exit_code=native.exit_code, duration_seconds=native.duration_seconds,
            # Claude's price table is not billing authority for a compatible
            # third-party provider. Preserve the raw CLI counter in native logs.
            token_usage=native.token_usage, total_cost_usd=(native.total_cost_usd if provider['id'] == 'anthropic' else None),
            tool_calls=native.tool_calls, files_read=native.files_read, files_written=native.files_written,
            raw_events=native.raw_events, stderr=native.stderr, effective_limits=dict(limits),
            workflow_hash=text_hash(workflow), spec_hash=text_hash(spec), model_calls=native.model_calls,
            errors=native.errors, timed_out=native.timed_out, completed=native.result_received)
