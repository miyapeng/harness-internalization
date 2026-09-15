"""Explicit scaffold registry; no model-name routing or online selection."""

def scaffold_for(name):
    if name == 'claude_code':
        from .claude_code import ClaudeCodeScaffold
        return ClaudeCodeScaffold()
    if name == 'codex':
        from .codex import CodexScaffold
        return CodexScaffold()
    if name == 'qwen_code':
        from .qwen_code import QwenCodeScaffold
        return QwenCodeScaffold()
    raise ValueError('unsupported scaffold: '+str(name))
