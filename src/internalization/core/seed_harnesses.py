"""Protected H0 provenance/binding for the three formal benchmark entrypoints.

This is not a new Harness type. Candidate workspaces cannot edit this registry
or the benchmark adapters. Resume never reloads seed source files.
"""
from pathlib import Path
import hashlib
import json

from .types import digest
from ..harness.revision import HarnessRevision

ROOT = Path(__file__).resolve().parents[3]
SEEDED_BENCHMARKS = ('alfworld', 'webshop', 'hotpotqa')
REGISTRY = ROOT / 'configs/seed_harnesses.json'


def bind_initial_seed(benchmark, revision):
    """Require the registered immutable H0 content for a fresh formal run."""
    if benchmark not in SEEDED_BENCHMARKS:
        return None
    if not isinstance(revision, HarnessRevision):
        raise ValueError('Fresh benchmark run requires its executable seed HarnessRevision')
    revision.files()
    registry = json.loads(REGISTRY.read_text())
    if registry.get('schema') != 1 or set(registry['seeds']) != set(SEEDED_BENCHMARKS):
        raise ValueError('Invalid protected seed registry')
    record = registry['seeds'][benchmark]
    if record['benchmark'] != benchmark or record['revision_hash'] != revision.version:
        raise ValueError(f'Fresh {benchmark} run requires matching seed_harnesses/{benchmark}; '
                         'generic/smoke/foreign/modified H0 is not a registered seed. Use --state to resume.')
    for path, expected in record['fixed_adapter_sha256'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Fixed seed adapter changed without a provenance update: {path}')
    # No live paths to mutable source directories are used to execute the run.
    return {**record, 'provenance_hash':digest(record), 'registry_hash':digest(registry)}
