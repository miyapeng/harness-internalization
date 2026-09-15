"""Shared subprocess lifecycle; native event schemas stay in each adapter."""
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from .base import ProposerSessionResult, ScaffoldPreflightError, text_hash
from .isolation import WorkspaceCapsule
from ...core.types import write_json


class NativeCLI:
    def __init__(self, *, which=shutil.which, run_command=subprocess.run,
                 process_factory=subprocess.Popen, capsule=None):
        self.which, self.run_command, self.process_factory = which, run_command, process_factory
        self.capsule = capsule if capsule is not None else WorkspaceCapsule(which=which, run_command=run_command)
        self._preflight = None

    def input_text(self, spec, workflow):
        return spec+'\n\nEXPLICIT WORKFLOW\n'+workflow

    def preflight(self):
        binary = self.which(self.executable)
        if not binary: raise ScaffoldPreflightError(self.executable+' executable not found on PATH')
        try:
            version = self.run_command([binary, '--version'], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
            help_text = self.run_command([binary, *self.help_args], capture_output=True, text=True, check=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            raise ScaffoldPreflightError(self.executable+' version/help preflight failed') from exc
        if any(flag not in help_text for flag in self.required_flags):
            raise ScaffoldPreflightError(self.executable+' lacks required native flags; no fallback')
        self.capsule.preflight()
        self._preflight = {'binary': binary, 'version': version, 'scaffold_id': self.scaffold_id}
        return dict(self._preflight)

    def run(self, *, workspace, model, spec, workflow, limits, output, provider):
        workspace, output = Path(workspace).resolve(strict=True), Path(output).resolve()
        if workspace.name != 'proposer_workspace': raise ValueError('Dedicated proposer_workspace required')
        preflight = self._preflight or self.preflight()
        self.validate_limits(limits)
        output.mkdir(parents=True, exist_ok=True)
        home = output/'isolated_home'; home.mkdir()
        command, additions = self.command(preflight['binary'], workspace, home, model, spec, workflow, limits, provider)
        wrapped = self.capsule.wrap(command, workspace, home)
        # No inherited plugins, project configuration, model credentials or proxies.
        env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'HOME': '/home/proposer', **additions}
        key = os.environ.get(provider['api_key_env'])
        if key: env[self.credential_env] = key
        result = ProposerSessionResult(self.scaffold_id, preflight['version'], provider, model,
            effective_limits=dict(limits), workflow_hash=text_hash(workflow), spec_hash=text_hash(spec))
        write_json(output/'invocation.json', {'command': wrapped, 'cwd': str(workspace),
            'provider': provider, 'limits': limits, 'environment_keys': sorted(env)})
        started = time.monotonic(); stdout = ''; stderr = ''
        try:
            process = self.process_factory(wrapped, cwd=str(workspace), env=env, stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
            try:
                stdout, stderr = process.communicate(self.input_text(spec, workflow),
                                                     timeout=limits['max_wall_time_s'])
            except subprocess.TimeoutExpired:
                result.timed_out = True
                if isinstance(process, subprocess.Popen):
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                else: process.kill()
                stdout, stderr = process.communicate()
            result.exit_code = process.returncode
        except OSError as exc:
            result.errors.append('CLI execution error: '+str(exc))
        finally:
            result.duration_seconds = time.monotonic()-started
            result.stderr = stderr or ''
            try:
                for line in (stdout or '').splitlines():
                    if not line.strip(): continue
                    event = json.loads(line)
                    if not isinstance(event, dict): raise ValueError('Event must be an object')
                    result.raw_events.append(event)
                self.parse_events(result)
                for value in result.token_usage.values():
                    if type(value) is not int or value < 0: raise ValueError('Invalid token usage')
                money = result.total_cost_usd
                if money is not None and (type(money) not in (int, float) or not math.isfinite(money) or money < 0):
                    raise ValueError('Invalid monetary cost')
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                result.errors.append('Invalid native events: '+str(exc))
                result.token_usage = {}; result.total_cost_usd = None
            if not result.session_id: result.errors.append('No native session ID')
            if not result.completed: result.errors.append('No successful terminal event')
            (output/'stdout.jsonl').write_text(stdout or '')
            (output/'stderr.log').write_text(result.stderr)
            write_json(output/'session.json', asdict(result))
        return result
