"""One explicit checkpoint/Harness/protocol pair for evolution, resume and evaluation."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from .serialization import harness_from_dict
from .types import digest
from ..harness.module import Harness
from ..harness.revision import HarnessRevision


def serialize_harness(harness):
    if isinstance(harness, HarnessRevision): return harness.to_dict()
    if not isinstance(harness, Harness): raise TypeError("Unsupported Harness type")
    return {"version": harness.version, "modules": [
        {"name": m.name, "version": m.version, "source": m.source} for m in harness.modules]}


@dataclass(frozen=True)
class AcceptedAgentState:
    checkpoint: str
    harness: Harness | HarnessRevision
    manifest_hash: str
    protocol: dict
    next_cycle: int | None = None

    def __post_init__(self):
        if not self.checkpoint or not self.manifest_hash or self.protocol.get("manifest_hash") != self.manifest_hash:
            raise ValueError("Accepted Agent needs matching checkpoint/manifest/protocol identity")
        if self.next_cycle is not None and (type(self.next_cycle) is not int or self.next_cycle < 0):
            raise ValueError("Invalid next_cycle")

    def to_dict(self):
        return {"agent_state_schema": 1, "checkpoint": self.checkpoint,
            "harness_revision": serialize_harness(self.harness), "manifest_hash": self.manifest_hash,
            "protocol": self.protocol, "protocol_hash": digest(self.protocol), "next_cycle": self.next_cycle}

    def check_manifest(self, manifest):
        manifest.validate()
        if self.manifest_hash != manifest.fingerprint: raise ValueError("Accepted state/manifest identity mismatch")

    def check_resume(self, manifest, config, policy, attribution_policy):
        from dataclasses import asdict
        self.check_manifest(manifest)
        if self.next_cycle is None: raise ValueError("Resume requires a completed cycle state, not an intermediate acceptance")
        if self.next_cycle >= config.cycles: raise ValueError("No unconsumed cycles remain in this protocol")
        expected = {"loop": asdict(config), "retirement": asdict(policy)}
        if "attribution" not in self.protocol:
            raise ValueError("Resume requires the recorded attribution policy; evaluation can still load this historical state")
        expected["attribution"] = asdict(attribution_policy)
        if isinstance(self.harness, HarnessRevision):
            expected.update(attribution=asdict(attribution_policy), harness_acceptance=asdict(attribution_policy))
        for name, value in expected.items():
            if name not in self.protocol or digest(self.protocol[name]) != digest(value):
                raise ValueError(f"Resume protocol mismatch: {name}")


def load_accepted_state(path, manifest, *, checkpoint=None, protocol_path=None):
    """Load canonical or historical state. Malformed/missing data never means baseline.

    Historical states must retain their adjacent experiment protocol (or an explicit
    protocol file). A legacy names-only deployment cannot reconstruct module code.
    """
    path = Path(path).resolve(strict=True)
    value = json.loads(path.read_text())
    if not isinstance(value, dict) or not isinstance(value.get("checkpoint"), str):
        raise ValueError("Accepted state must contain a checkpoint; baseline must be explicit")
    schema = value.get("agent_state_schema")
    if schema is not None and (type(schema) is not int or schema != 1): raise ValueError("Unsupported accepted Agent state schema")
    if "harness_revision" in value:
        raw = value["harness_revision"]
        if isinstance(raw, dict) and raw.get("format") == "code_revision_v1":
            raw = dict(raw)
            if not Path(raw["path"]).is_absolute(): raw["path"] = str(path.parent/raw["path"])
    elif "harness_version" in value and "active_modules" in value:
        modules = value["active_modules"]
        if not isinstance(modules, list) or any(not isinstance(m, dict) or "source" not in m for m in modules):
            raise ValueError("Legacy deployment contains module names only; use its accepted cycle state.json with source")
        raw = {"version": value["harness_version"], "modules": modules}
    else: raise ValueError("Accepted state lacks a runnable Harness; no empty-Harness fallback")
    try: harness = harness_from_dict(raw)
    except (KeyError, TypeError, AttributeError) as exc: raise ValueError("Malformed accepted Harness") from exc
    location = Path(value["checkpoint"])
    if not location.is_absolute(): location = path.parent/location
    location = location.resolve(strict=True)
    if not location.is_dir(): raise ValueError("Accepted checkpoint must be an existing directory")
    if checkpoint is not None and Path(checkpoint).resolve(strict=True) != location:
        raise ValueError("Accepted state checkpoint mismatch")

    protocol = value.get("protocol")
    protocol_origin = None
    if protocol_path is not None:
        supplied = json.loads(Path(protocol_path).read_text())
        if protocol is not None and digest(protocol) != digest(supplied): raise ValueError("Conflicting protocol identities")
        protocol = supplied
        protocol_origin = Path(protocol_path)
    if protocol is None and schema is None:
        paths = (path.parent/"protocol.json", path.parent.parent/"protocol.json")
        found = next((p for p in paths if p.is_file()), None)
        if found:
            protocol = json.loads(found.read_text())
            protocol_origin = found
    if not isinstance(protocol, dict) or not protocol.get("manifest_hash"):
        raise ValueError("Accepted state needs its experiment protocol and manifest identity")
    if schema is None and "attribution" not in protocol and protocol_origin is not None:
        attribution_file = protocol_origin.parent/"attribution_policy.json"
        if attribution_file.is_file():
            protocol = {**protocol, "attribution":json.loads(attribution_file.read_text())}
    if schema == 1 and any(k not in value for k in ("protocol", "protocol_hash", "manifest_hash", "next_cycle")):
        raise ValueError("Incomplete canonical accepted Agent state")
    if "protocol_hash" in value and value["protocol_hash"] != digest(protocol):
        raise ValueError("Accepted protocol hash mismatch")
    manifest_hash = value.get("manifest_hash", protocol["manifest_hash"])
    if "next_cycle" in value:
        next_cycle = value["next_cycle"]
        if "cycle" in value and next_cycle != value["cycle"]+1:
            raise ValueError("Accepted cycle/next_cycle mismatch")
    elif path.name == "accepted_harness.json": next_cycle = None
    elif "cycle" in value: next_cycle = value["cycle"] + 1
    elif re.fullmatch(r"cycle_\d+", path.parent.name): next_cycle = int(path.parent.name.split("_")[-1]) + 1
    elif path.name == "deployment.json": next_cycle = protocol.get("loop", {}).get("cycles")
    else: next_cycle = None
    agent = AcceptedAgentState(str(location), harness, manifest_hash, protocol, next_cycle)
    agent.check_manifest(manifest)
    return agent


def add_evaluation_agent_arguments(parser):
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--state", type=Path, help="Accepted state/deployment with checkpoint, Harness and protocol identity")
    selection.add_argument("--baseline", action="store_true", help="Explicitly evaluate an initial empty Harness")
    parser.add_argument("--checkpoint", type=Path, help="Required for baseline; optional consistency check for --state")
    parser.add_argument("--protocol", type=Path, help="Protocol sidecar for a relocated historical state")


def evaluation_agent(args, manifest):
    if args.state: return load_accepted_state(args.state, manifest, checkpoint=args.checkpoint, protocol_path=args.protocol)
    if not args.baseline or args.checkpoint is None: raise ValueError("Explicit baseline requires --checkpoint")
    if args.protocol: raise ValueError("--protocol is only for an accepted state")
    checkpoint = args.checkpoint.resolve(strict=True)
    if not checkpoint.is_dir(): raise ValueError("Checkpoint must be a directory")
    return AcceptedAgentState(str(checkpoint), Harness(), manifest.fingerprint,
        {"mode": "explicit_baseline", "manifest_hash": manifest.fingerprint})
