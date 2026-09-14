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
        validate_config(config,files,self.policy)
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

    def bind_patch(self, parent, changes):
        """Bind model-written path/content changes to a verified immutable parent.

        Hashes are host metadata, never a responsibility of the proposal model.
        apply() independently rechecks the tree and every precondition afterwards.
        """
        if parent.policy != self.policy: raise ValueError("Candidate cannot change workspace permissions")
        files, seen, patch = parent.files(), set(), []
        for change in changes:
            if not isinstance(change,dict) or set(change) != {"path", "content"}:
                raise ValueError("Proposed edit must contain only path and content; hashes are host-generated")
            path = self.policy.validate_path(change["path"])
            if path in seen: raise ValueError("Repeated patch path")
            seen.add(path)
            content = change["content"]
            if content is not None and not isinstance(content,str): raise ValueError("Patch content must be text")
            if content is None and path not in files: raise ValueError("Cannot remove missing file")
            patch.append(FileEdit(path,text_hash(files[path]) if path in files else None,content))
        return tuple(patch)

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


def validate_config(config,files,policy):
    if config.get("schema")==1 and set(config)=={"schema","entrypoint","supervision"}:
        if config["supervision"] is not None: validate_entrypoint(config["supervision"],files,policy)
    elif config.get("schema")==2 and set(config)=={"schema","entrypoint","controls","composition"}:
        if config["composition"] not in ("independent_suffix","sequential_suffix"):
            raise ValueError("Unsupported control composition")
        if not isinstance(config["controls"],list): raise ValueError("Controls must be an ordered list")
        seen=set()
        for control in config["controls"]:
            if not isinstance(control,dict) or set(control)!={"id","entrypoint","enabled"}:
                raise ValueError("Control requires id, entrypoint and enabled")
            cid=control["id"]
            if not isinstance(cid,str) or not cid or len(cid)>128 or not all(c.isalnum() or c in "_-" for c in cid):
                raise ValueError("Invalid control ID")
            if cid in seen: raise ValueError("Duplicate control ID")
            seen.add(cid)
            if type(control["enabled"]) is not bool: raise ValueError("Control enabled must be boolean")
            validate_entrypoint(control["entrypoint"],files,policy)
    else: raise ValueError("Unsupported Harness interface configuration")
    validate_entrypoint(config["entrypoint"],files,policy)


@dataclass(frozen=True)
class InternalizationTarget:
    full_revision: HarnessRevision
    reduced_revision: HarnessRevision
    removed_behavior: str
    supervision_adapter: str
    target_control_id: str | None = None

    @property
    def target_id(self): return digest(self.to_dict())

    def to_dict(self):
        value={"full_revision":self.full_revision.to_dict(),"reduced_revision":self.reduced_revision.to_dict(),
                "removed_behavior":self.removed_behavior,"supervision_adapter":self.supervision_adapter}
        if self.target_control_id is not None: value["target_control_id"]=self.target_control_id
        return value

    @classmethod
    def from_dict(cls,value):
        return cls(HarnessRevision.from_dict(value["full_revision"]),HarnessRevision.from_dict(value["reduced_revision"]),
                   value["removed_behavior"],value["supervision_adapter"],value.get("target_control_id"))

    @classmethod
    def from_control(cls,store,full,target_control_id,removed_behavior):
        config=full.config
        if config["schema"]!=2: raise ValueError("Named target requires schema 2")
        if config["composition"]!="independent_suffix":
            raise ValueError("unsupported: dependent control composition cannot reuse non-target context")
        if not isinstance(removed_behavior,str) or not removed_behavior.strip():
            raise ValueError("Target must identify its removed behavior")
        selected=[c for c in config["controls"] if c["id"]==target_control_id and c["enabled"]]
        if len(selected)!=1: raise ValueError("Target must select one enabled control ID")
        reduced_config={**config,"controls":[{**c,"enabled":False} if c["id"]==target_control_id else c for c in config["controls"]]}
        reduced=store.apply(full,store.bind_patch(full,[{"path":"config/harness.json","content":json.dumps(reduced_config)}]))
        target=cls(full,reduced,removed_behavior,selected[0]["entrypoint"],target_control_id)
        target.validate_structure()
        return target

    @classmethod
    def from_supervision(cls,store,full,removed_behavior,supervision_adapter):
        """Construct the supported reduction on the host; retain all shared code."""
        config = full.config
        if not isinstance(removed_behavior,str) or not removed_behavior.strip():
            raise ValueError("Target must identify its removed behavior")
        if supervision_adapter is None or config["supervision"] != supervision_adapter:
            raise ValueError("unsupported: target must select the registered supervision hook")
        changes = [{"path":"config/harness.json",
                    "content":json.dumps({**config,"supervision":None})}]
        reduced = store.apply(full,store.bind_patch(full,changes))
        target = cls(full,reduced,removed_behavior,supervision_adapter)
        target.validate_structure()
        return target

    def validate_structure(self):
        plus, minus = self.full_revision, self.reduced_revision
        full_files, reduced_files = plus.files(), minus.files()
        if plus.policy != minus.policy or not self.removed_behavior.strip():
            raise ValueError("Target cannot change permissions and must identify its removed behavior")
        cfg_plus, cfg_minus = plus.config, minus.config
        if self.target_control_id is not None:
            if cfg_plus["schema"]!=2 or cfg_plus["composition"]!="independent_suffix":
                raise ValueError("unsupported: dependent control composition cannot reuse non-target context")
            selected=[c for c in cfg_plus["controls"] if c["id"]==self.target_control_id and c["enabled"]]
            if len(selected)!=1 or selected[0]["entrypoint"]!=self.supervision_adapter:
                raise ValueError("Target must select its enabled registered control")
            expected={**cfg_plus,"controls":[{**c,"enabled":False} if c["id"]==self.target_control_id else c for c in cfg_plus["controls"]]}
        else:
            if cfg_plus.get("schema")!=1 or cfg_plus["supervision"] != self.supervision_adapter or cfg_minus.get("supervision") is not None:
                raise ValueError("unsupported: target must bypass the actually executed supervision hook")
            expected={**cfg_plus,"supervision":None}
        # v1 bridge supports removal of an internal computation hook, not arbitrary tool/context changes.
        # All non-target code, prompts, registrations and config must remain byte-identical.
        if ({k:v for k,v in full_files.items() if k!="config/harness.json"} !=
            {k:v for k,v in reduced_files.items() if k!="config/harness.json"} or
            expected != cfg_minus):
            raise ValueError("unsupported: reduced revision changes shared runtime/tool/context code")
        validate_entrypoint(self.supervision_adapter,full_files,plus.policy)
