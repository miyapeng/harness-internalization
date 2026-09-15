"""Strict, explicit profiles. Resolved records never re-read a mutable profile file."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

from .materialization import SPEC_PATH
from .scaffolds.base import text_hash, ScaffoldPreflightError

PROFILE_ROOT = Path(__file__).resolve().parents[3] / 'configs/proposers'
WORKFLOW_ROOT = Path(__file__).with_name('workflows')
SCAFFOLDS = {'claude_code': 'anthropic', 'codex': 'responses', 'qwen_code': 'openai'}
FIELDS = {'profile_id', 'scaffold', 'provider', 'model', 'limits', 'workflow', 'cli_version', 'experiment_mode'}
RESOLVED_FIELDS = FIELDS | {'resolved_schema', 'spec_hash', 'workflow_hash'}


def workflow_text(name):
    if name not in SCAFFOLDS: raise ValueError('unsupported workflow: '+str(name))
    return (WORKFLOW_ROOT/(name+'.md')).read_text()


def resolve_profile(value, *, profile_root=None):
    if not isinstance(value, dict): raise ValueError('proposer must be an object')
    if set(value) == {'profile'}:
        name = value['profile']
        if not isinstance(name, str) or not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
            raise ValueError('Invalid explicit proposer profile ID')
        root = Path(profile_root) if profile_root is not None else PROFILE_ROOT
        if profile_root is None and not root.is_dir(): root = Path(sys.prefix)/'share/harness-internalization/proposers'
        value = json.loads((root/(name+'.json')).read_text())
        if value.get('profile_id') != name: raise ValueError('Profile filename/ID mismatch')
    value = deepcopy(value)
    resolved = 'resolved_schema' in value
    if set(value) != (RESOLVED_FIELDS if resolved else FIELDS):
        raise ValueError('Unknown or missing proposer profile fields')
    name = value['scaffold']
    if not isinstance(name, str) or name not in SCAFFOLDS: raise ValueError('unsupported scaffold: '+str(name))
    if not isinstance(value['workflow'], str) or value['workflow'] not in SCAFFOLDS: raise ValueError('unsupported workflow')
    if value['experiment_mode'] not in ('native', 'fixed'): raise ValueError('Unknown scaffold experiment mode')
    provider = value['provider']
    if not isinstance(provider, dict) or set(provider) != {'id', 'protocol', 'base_url', 'api_key_env'}:
        raise ValueError('Provider requires id/protocol/base_url/api_key_env, never credentials')
    if provider['protocol'] != SCAFFOLDS[name]: raise ValueError('Provider protocol incompatible with scaffold')
    if not isinstance(provider['id'], str) or not provider['id']: raise ValueError('Provider ID required')
    endpoint = urlparse(provider['base_url'])
    if endpoint.scheme != 'https' or not endpoint.hostname or endpoint.username or endpoint.password or endpoint.query or endpoint.fragment:
        raise ValueError('Explicit HTTPS provider URL required, without secrets')
    if not isinstance(provider['api_key_env'], str) or not re.fullmatch(r'[A-Z][A-Z0-9_]*', provider['api_key_env']):
        raise ValueError('Credential environment variable name required')
    for key in ('model', 'profile_id', 'cli_version'):
        v = value[key]
        # Null model/version is a visible template, rejected before any real session.
        if v is None and key in ('model', 'cli_version'): continue
        if not isinstance(v, str) or not v.strip() or '\n' in v:
            raise ValueError('Invalid profile '+key)
    if value['model'] is not None and (any(c.isspace() for c in value['model']) or value['model'].endswith('-latest') or value['model'].startswith('-') or value['model'] in ('sonnet','opus','haiku','default','auto','latest')):
        raise ValueError('Exact provider model ID required')
    limits = value['limits']
    if not isinstance(limits, dict) or set(limits) != {'max_turns', 'max_wall_time_s', 'max_tool_calls'}:
        raise ValueError('Explicit turn, wall-time and tool limits required')
    for k, v in limits.items():
        if v is None and k in ('max_tool_calls', 'max_turns'): continue
        if type(v) is not int or v < 1: raise ValueError('Positive integer required: limits.'+k)
    hashes = {'spec_hash': text_hash(SPEC_PATH.read_text()), 'workflow_hash': text_hash(workflow_text(value['workflow']))}
    if resolved and (value['resolved_schema'] != 1 or any(value[k] != v for k, v in hashes.items())):
        raise ValueError('Resolved proposer spec/workflow changed; cannot silently resume')
    return {**value, 'resolved_schema': 1, **hashes}


def verify_runtime(profile, preflight):
    if profile['model'] is None:
        raise ScaffoldPreflightError('Profile needs the exact provider model ID; no model substitution')
    if profile['cli_version'] is None:
        raise ScaffoldPreflightError('Profile needs an exact CLI version before running; no floating runtime')
    if preflight['version'] != profile['cli_version']:
        raise ScaffoldPreflightError('CLI version differs from resolved proposer protocol')


def resume_profile(requested, recorded):
    """Bind a matching reference to the state snapshot, without reading its file.

    A different reference or explicit different resolved settings is a protocol
    change and must start a new experiment. Non-proposer resume checks stay intact.
    """
    saved = resolve_profile(recorded)
    if requested is None: return saved
    if set(requested) == {'profile'}:
        if requested['profile'] != saved['profile_id']:
            raise ValueError('Resume proposer profile_id mismatch')
        return saved
    if resolve_profile(requested) != saved:
        raise ValueError('Resume resolved proposer profile mismatch')
    return saved


def suggest_default_profile(model_family):
    """Development hint only; never called by experiment resolution or execution."""
    return {'claude': 'claude-sonnet5_claude-code_v1', 'gpt': 'gpt-codex_codex_v1',
            'qwen': 'qwen-coder_qwen-code_v1', 'deepseek': 'deepseek-v41-flash_claude-code_v1'}[model_family.lower()]
