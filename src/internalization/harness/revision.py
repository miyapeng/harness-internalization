"""Content-addressed runnable Harness trees, separate from the experiment implementation."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import uuid

from ..core.types import digest, write_json


def text_hash(text): return hashlib.sha256(text.encode()).hexdigest()


@dataclass(frozen=True)
class WorkspacePolicy:
    editable_prefixes: tuple[str, ...] = ("agent/", "prompts/", "tools/", "controls/", "config/")
    capabilities: tuple[str, ...] = ("model", "environment")
    max_files: int = 64
    max_bytes: int = 262144

    def validate_path(self, value):
        path = PurePosixPath(value)
        if (not isinstance(value, str) or path.is_absolute() or str(path) != value or
            any(part in ("", ".", "..") or part.startswith(".") for part in path.parts) or
            not any(value.startswith(prefix) for prefix in self.editable_prefixes) or
            path.suffix not in (".py", ".json", ".txt", ".md")):
            raise ValueError(f"Patch outside configured Harness workspace: {value}")
        return value


@dataclass(frozen=True)
class FileEdit:
    path: str
    before_hash: str | None
    content: str | None


@dataclass(frozen=True)
class HarnessRevision:
    version: str
    path: str
    policy: WorkspacePolicy

    def files(self):
        root = Path(self.path).resolve(strict=True)
        files = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink(): raise ValueError("Symlink in Harness revision")
            if path.is_file():
                name = self.policy.validate_path(path.relative_to(root).as_posix())
                files[name] = path.read_text()
        if digest({"files":files,"policy":asdict(self.policy)}) != self.version:
            raise ValueError("Harness revision content changed")
        return files

    @property
    def config(self): return json.loads(self.files()["config/harness.json"])

    def to_dict(self): return {"format":"code_revision_v1", **asdict(self)}

    @classmethod
    def from_dict(cls, value):
        policy = dict(value["policy"])
        for k in ("editable_prefixes", "capabilities"): policy[k] = tuple(policy[k])
        result = cls(value["version"], value["path"], WorkspacePolicy(**policy))
        result.files()
        return result

    def without(self, target):
        if not isinstance(target, InternalizationTarget) or target.full_revision.version != self.version:
            raise ValueError("An explicit matching InternalizationTarget is required")
        target.validate_structure()
        return target.reduced_revision


class RevisionStore:
    def __init__(self, root, policy=WorkspacePolicy()):
        self.root, self.policy = Path(root).resolve(), policy
        self.root.mkdir(parents=True, exist_ok=True)

    def snapshot(self, files):
        files = dict(files)
        if len(files)>self.policy.max_files or sum(len(v.encode()) for v in files.values())>self.policy.max_bytes:
            raise ValueError("Harness tree exceeds fixed workspace limits")
        for path, source in files.items():
            self.policy.validate_path(path)
            if path.endswith(".py"): ast.parse(source, filename=path)  # Never exec in host.
            if path.endswith(".json"): json.loads(source)
        config = json.loads(files["config/harness.json"])
        if set(config) != {"schema", "entrypoint", "supervision"} or config["schema"] != 1:
            raise ValueError("Unsupported Harness interface configuration")
        validate_entrypoint(config["entrypoint"], files, self.policy)
        if config["supervision"] is not None: validate_entrypoint(config["supervision"],files,self.policy)
        version = digest({"files":files,"policy":asdict(self.policy)})
        # Always construct in a fresh directory. Existing revisions are never patched in place.
        folder = self.root/(version+"-"+uuid.uuid4().hex[:12])
        folder.mkdir()
        for name, source in files.items():
            path = folder/name
            path.parent.mkdir(parents=True,exist_ok=True)
            with path.open("x") as file: file.write(source)
            path.chmod(0o444)
        result = HarnessRevision(version,str(folder),self.policy)
        result.files()
        return result

    def import_directory(self, directory):
        directory = Path(directory).resolve(strict=True)
        files = {}
        for path in directory.rglob("*"):
            if path.is_symlink(): raise ValueError("Harness import cannot contain symlinks")
            if path.is_file(): files[path.relative_to(directory).as_posix()] = path.read_text()
        return self.snapshot(files)

    def apply(self, parent, patch):
        if parent.policy != self.policy: raise ValueError("Candidate cannot change workspace permissions")
        files, seen = parent.files(), set()
        for edit in patch:
            self.policy.validate_path(edit.path)
            if edit.path in seen: raise ValueError("Repeated patch path")
            seen.add(edit.path)
            actual = text_hash(files[edit.path]) if edit.path in files else None
            if actual != edit.before_hash: raise ValueError("Patch base hash mismatch")
            if edit.content is None:
                if edit.path not in files: raise ValueError("Cannot remove missing file")
                del files[edit.path]  # Only the new in-memory tree; parent/user files are untouched.
            else:
                if not isinstance(edit.content,str): raise ValueError("Patch content must be text")
                files[edit.path] = edit.content
        return self.snapshot(files)


def validate_entrypoint(entrypoint, files, policy):
    if not isinstance(entrypoint,str) or entrypoint.count(":") != 1: raise ValueError("Expected path.py:function")
    path, function = entrypoint.split(":")
    policy.validate_path(path)
    if path not in files or not path.endswith(".py") or not function.isidentifier():
        raise ValueError("Missing executable Harness entrypoint")


@dataclass(frozen=True)
class InternalizationTarget:
    full_revision: HarnessRevision
    reduced_revision: HarnessRevision
    removed_behavior: str
    supervision_adapter: str

    @property
    def target_id(self): return digest(self.to_dict())

    def to_dict(self):
        return {"full_revision":self.full_revision.to_dict(),"reduced_revision":self.reduced_revision.to_dict(),
                "removed_behavior":self.removed_behavior,"supervision_adapter":self.supervision_adapter}

    @classmethod
    def from_dict(cls,value):
        return cls(HarnessRevision.from_dict(value["full_revision"]),HarnessRevision.from_dict(value["reduced_revision"]),
                   value["removed_behavior"],value["supervision_adapter"])

    def validate_structure(self):
        plus, minus = self.full_revision, self.reduced_revision
        full_files, reduced_files = plus.files(), minus.files()
        if plus.policy != minus.policy or not self.removed_behavior.strip():
            raise ValueError("Target cannot change permissions and must identify its removed behavior")
        cfg_plus, cfg_minus = plus.config, minus.config
        if cfg_plus["supervision"] != self.supervision_adapter or cfg_minus["supervision"] is not None:
            raise ValueError("unsupported: target must bypass the actually executed supervision hook")
        # v1 bridge supports removal of an internal computation hook, not arbitrary tool/context changes.
        # All non-target code, prompts, registrations and config must remain byte-identical.
        if ({k:v for k,v in full_files.items() if k!="config/harness.json"} !=
            {k:v for k,v in reduced_files.items() if k!="config/harness.json"} or
            {**cfg_plus,"supervision":None} != cfg_minus):
            raise ValueError("unsupported: reduced revision changes shared runtime/tool/context code")
        validate_entrypoint(self.supervision_adapter,full_files,plus.policy)
