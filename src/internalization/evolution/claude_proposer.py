"""Compatibility constructor for existing injected Claude tests; production uses the shared host."""
from .proposer_host import WorkspaceProposalHost, canonical_diff
from .proposer_profiles import resolve_profile
from .scaffolds.claude_code import ClaudeCodeScaffold


class ClaudeCodeProposer(WorkspaceProposalHost):
    def __init__(self, store, *, backend='claude_code', model='claude-sonnet-5', max_turns=12, runner=None):
        if backend != 'claude_code': raise ValueError('Use an explicit proposer profile for other scaffolds')
        profile = resolve_profile({'profile': 'claude-sonnet5_claude-code_v1'})
        profile['model'] = model
        profile['limits']['max_turns'] = max_turns
        super().__init__(store, profile=profile, scaffold=ClaudeCodeScaffold(runner=runner),
                         compatibility=runner is not None)
