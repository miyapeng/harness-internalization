"""One Claude editing session -> canonical edits -> existing host materialization."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
import sys

from .claude_code import ClaudeCodeRunner, ClaudeCodePreflightError
from .materialization import (SPEC_PATH, materialize_candidates, public_execution_trace,
                            validate_search_input)
from ..core.types import Cost, write_json


def skill_path():
    source = Path(__file__).resolve().parents[3]/'.claude/skills/harness-internalization-proposer/SKILL.md'
    installed = Path(sys.prefix)/'share/harness-internalization/SKILL.md'
    if source.is_file(): return source
    if installed.is_file(): return installed
    raise ClaudeCodePreflightError('Explicit proposer skill resource is missing; reinstall the project')


def read_tree(root, policy):
    """Read text-only regular trees without following symlinks or hardlinks."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir(): raise ValueError('Candidate workspace must be a real directory')
    files = {}; total = 0
    for directory, folders, names in os.walk(root, followlinks=False):
        for name in folders:
            if (Path(directory)/name).is_symlink(): raise ValueError('Symlink directory forbidden')
        for name in names:
            path = Path(directory)/name
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise ValueError('Only unlinked regular candidate files are allowed')
            relative = path.relative_to(root).as_posix(); policy.validate_path(relative)
            total += info.st_size
            if len(files) >= policy.max_files or total > policy.max_bytes: raise ValueError('Candidate workspace exceeds policy limits')
            files[relative] = path.read_text(encoding='utf-8')
    return files


def canonical_diff(parent_files, candidate_files):
    return [{'path': path, 'content': candidate_files.get(path)}
        for path in sorted(parent_files.keys() | candidate_files.keys())
        if parent_files.get(path) != candidate_files.get(path)]


def public_history_files(history):
    """Keep search feedback and candidate code; remove host snapshot filesystem paths."""
    def strip(value):
        if isinstance(value, list): return [strip(v) for v in value]
        if not isinstance(value, dict): return value
        # Only executable revision descriptors contain format=code_revision_v1.
        return {k: strip(v) for k, v in value.items()
            if not (k == 'path' and value.get('format') == 'code_revision_v1')}
    return strip(list(history))


class ClaudeCodeProposer:
    def __init__(self, store, *, backend='claude_code', model='claude-sonnet-5', max_turns=12, runner=None):
        if backend != 'claude_code': raise ValueError('Production proposer backend must be claude_code')
        if not isinstance(model, str) or not model.startswith('claude-') or model.endswith('-latest') or any(c.isspace() for c in model):
            raise ValueError('Proposer requires an exact Claude model ID, not a CLI alias')
        if type(max_turns) is not int or max_turns <= 0: raise ValueError('Positive proposer.max_turns required')
        self.store, self.model, self.max_turns = store, model, max_turns
        self.runner = runner if runner is not None else ClaudeCodeRunner()
        self.last_cost = Cost()

    def propose(self, request):
        if request.count != 2: raise ValueError('Claude proposer requires exactly 2 candidates in one session')
        history = validate_search_input(request)
        parent_files = request.harness.files()  # Verify before copying, and again after session.
        request.output.mkdir(parents=True, exist_ok=True)
        try: preflight = self.runner.preflight()
        except ClaudeCodePreflightError as exc:
            write_json(request.output/'preflight_error.json', {'error': str(exc), 'proposal_sessions': 0})
            write_json(request.output/'cost.json', asdict(self.last_cost))
            raise
        workspace = request.output/'claude_workspace'; workspace.mkdir()
        for name in ('parent', 'candidate_0', 'candidate_1'):
            for relative, content in parent_files.items():
                path = workspace/name/relative; path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding='utf-8')
        write_json(workspace/'evidence/scores.json', [asdict(s) for s in request.scores])
        write_json(workspace/'evidence/workspace_policy.json', asdict(request.harness.policy))
        for index, trace in enumerate(request.trajectories):
            write_json(workspace/f'evidence/trajectory_{index}.json', public_execution_trace(trace))
        write_json(workspace/'history.json', public_history_files(history))
        spec, skill = SPEC_PATH.read_text(), skill_path().read_text()
        (workspace/'PROPOSER_SPEC.md').write_text(spec)
        protected = {p.relative_to(workspace).as_posix(): p.read_bytes() for p in workspace.rglob('*')
            if p.is_file() and p.relative_to(workspace).parts[0] not in ('candidate_0', 'candidate_1')}
        for relative in protected: (workspace/relative).chmod(0o444)
        session = self.runner.run(model=self.model, max_turns=self.max_turns, cwd=workspace,
            output=request.output/'claude_session', spec=spec, skill=skill, preflight=preflight)
        self.last_cost = session.cost
        write_json(request.output/'cost.json', asdict(self.last_cost))
        write_json(request.output/'proposer_protocol.json', {'backend': 'claude_code', 'proposal_sessions': 1,
            'candidate_count': 2, 'parent_revision': request.harness.version, 'model': self.model,
            'max_turns': self.max_turns, 'cli_version': session.cli_version, 'session_id': session.session_id,
            'allowed_tools': list(session.allowed_tools), 'spec_hash': session.spec_hash, 'skill_hash': session.skill_hash,
            'total_cost_usd': session.total_cost_usd, 'succeeded': session.succeeded})
        if not session.succeeded: raise RuntimeError('Claude proposer session failed; see claude_session/session.json')
        if request.harness.files() != parent_files: raise ValueError('Parent revision changed during proposal')
        if read_tree(workspace/'parent', request.harness.policy) != parent_files:
            raise ValueError('Protected parent workspace changed')
        for relative, original in protected.items():
            p = workspace/relative
            if p.is_symlink() or not p.is_file() or p.read_bytes() != original:
                raise ValueError('Protected proposer input changed: '+relative)
        for name in ('parent', 'evidence'):
            directory = workspace/name
            if directory.is_symlink(): raise ValueError('Protected directory cannot be a symlink')
            for path in directory.rglob('*'):
                if path.is_symlink() or (path.is_file() and path.relative_to(workspace).as_posix() not in protected):
                    raise ValueError('Unexpected protected input entry')
        if {p.name for p in workspace.iterdir()} != {'parent', 'candidate_0', 'candidate_1', 'evidence', 'history.json', 'PROPOSER_SPEC.md', 'proposal.json'}:
            raise ValueError('Unexpected or missing proposer workspace entry')
        path = workspace/'proposal.json'
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode) or path.stat().st_nlink != 1 or path.stat().st_size > request.harness.policy.max_bytes:
            raise ValueError('Invalid proposal metadata file')
        metadata = json.loads(path.read_text())
        if not isinstance(metadata, dict) or set(metadata) != {'candidates'} or not isinstance(metadata['candidates'], list) or len(metadata['candidates']) != 2:
            raise ValueError('proposal.json must contain exactly two candidates')
        rows = []
        for index, row in enumerate(metadata['candidates']):
            try:
                if not isinstance(row, dict) or set(row) != {'workspace', 'rationale', 'evidence_refs', 'internalization'} or row['workspace'] != f'candidate_{index}':
                    raise ValueError('Invalid structured workspace metadata')
                files = read_tree(workspace/row['workspace'], request.harness.policy)
                rows.append({k: v for k, v in row.items() if k != 'workspace'} | {'patch': canonical_diff(parent_files, files)})
            except (ValueError, OSError) as exc:
                write_json(request.output/f'candidate_{index}_workspace_error.json', {'reason': str(exc)})
                rows.append({'workspace_error': str(exc)})
        # One validation/materialization implementation shared with API regression fixtures.
        return materialize_candidates(self.store, request, rows)
