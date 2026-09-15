"""Session infrastructure only. This interface knows no benchmark or candidate types."""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
import hashlib

from ...core.types import Cost


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


class ScaffoldPreflightError(RuntimeError):
    pass


@dataclass
class ProposerSessionResult:
    scaffold_id: str
    scaffold_version: str
    provider: dict
    model: str
    session_id: str = ''
    exit_code: int | None = None
    duration_seconds: float = 0.0
    token_usage: dict = field(default_factory=dict)
    total_cost_usd: float | None = None
    tool_calls: list = field(default_factory=list)
    files_read: dict | None = None
    files_written: dict | None = None
    raw_events: list = field(default_factory=list)
    stderr: str = ''
    effective_limits: dict = field(default_factory=dict)
    workflow_hash: str = ''
    spec_hash: str = ''
    model_calls: int | None = None
    errors: list = field(default_factory=list)
    timed_out: bool = False
    completed: bool = False

    @property
    def succeeded(self):
        return self.exit_code == 0 and self.completed and not self.errors and not self.timed_out

    @property
    def cost(self):
        # Existing ledger has numeric counters, not nullable money fields. Session
        # JSON remains authoritative for unavailable values; no USD is invented.
        usage = self.token_usage
        inputs = usage.get('input_tokens', 0) + usage.get('cache_creation_input_tokens', 0)
        if self.scaffold_id == 'claude_code': inputs += usage.get('cache_read_input_tokens', 0)
        return Cost(inputs, usage.get('output_tokens', 0), self.model_calls or 0,
                    self.model_calls or 0, len(self.tool_calls), self.duration_seconds)


class ProposerScaffold(Protocol):
    scaffold_id: str

    def preflight(self) -> dict: ...

    def run(self, *, workspace: Path, model: str, spec: str, workflow: str,
            limits: dict, output: Path, provider: dict) -> ProposerSessionResult: ...
