"""Minimal Linux filesystem capsule for native coding CLIs, never the repo root.

The CLI needs provider networking. Tool networking is separately disabled by the
native Codex sandbox or Qwen's deny-all-except-file-tools hook. This is not a
general network sandbox: no shell/network/MCP tools may bypass those controls.
No container images, runtime dependencies or benchmark files are downloaded.
"""
from pathlib import Path
import shutil
import subprocess

from .base import ScaffoldPreflightError


class WorkspaceCapsule:
    def __init__(self, *, which=shutil.which, run_command=subprocess.run):
        self.which, self.run_command = which, run_command
        self.binary = None

    def base_command(self):
        command = [self.binary, '--unshare-all', '--share-net', '--die-with-parent',
                   '--new-session', '--cap-drop', 'ALL', '--ro-bind', '/usr', '/usr']
        for name in ('bin', 'sbin', 'lib', 'lib64'):
            path = Path('/')/name
            if path.is_symlink(): command += ['--symlink', str(path.readlink()), str(path)]
            elif path.is_dir(): command += ['--ro-bind', str(path), str(path)]
        command += ['--dir', '/etc']
        for name in ('ssl/certs', 'resolv.conf', 'hosts', 'nsswitch.conf', 'ld.so.cache'):
            path = Path('/etc')/name
            if path.exists(): command += ['--ro-bind', str(path), str(path)]
        return command + ['--tmpfs', '/tmp', '--proc', '/proc', '--dev', '/dev', '--tmpfs', '/home']

    def preflight(self):
        self.binary = self.which('bwrap')
        if not self.binary:
            raise ScaffoldPreflightError('bwrap executable required for native scaffold filesystem isolation; no host fallback')
        try:
            self.run_command(self.base_command()+['--', '/usr/bin/true'], check=True,
                             capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError) as exc:
            raise ScaffoldPreflightError('Native scaffold namespace isolation unavailable; no host fallback') from exc

    def wrap(self, command, workspace, home):
        if self.binary is None: raise ScaffoldPreflightError('Capsule preflight has not passed')
        workspace, home = Path(workspace).resolve(strict=True), Path(home).resolve(strict=True)
        if not Path(command[0]).resolve().is_relative_to('/usr'):
            raise ScaffoldPreflightError('Native CLI must be installed under /usr; ambient shell wrappers are forbidden')
        # Root workspace is read-only; only the two candidate trees and metadata
        # can be mutated. No experiment directory, model, repo or host home mounted.
        metadata = workspace/'proposal.json'
        if not metadata.exists(): metadata.write_text('')
        result = self.base_command()+['--ro-bind', str(workspace), str(workspace),
                                     '--bind', str(home), '/home/proposer']
        for name in ('candidate_0', 'candidate_1', 'proposal.json'):
            p = str(workspace/name); result += ['--bind', p, p]
        # Runtime settings cannot be rewritten by a model-generated command.
        for name in ('.codex/config.toml', '.qwen/settings.json', 'guard.py', 'file_guard.py'):
            p = home/name
            if p.exists(): result += ['--ro-bind', str(p), '/home/proposer/'+name]
        return result+['--chdir', str(workspace), '--', *command]
