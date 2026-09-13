from dataclasses import dataclass
from ..core.types import digest
from ..harness.module import HarnessModule
from ..harness.revision import HarnessRevision, FileEdit


@dataclass(frozen=True)
class Candidate:
    source: str
    parent: str
    version: int
    index: int

    @property
    def content_hash(self): return digest(self.source)

    @property
    def candidate_id(self): return digest([self.parent, self.version, self.index, self.content_hash])

    def module(self): return HarnessModule.from_source(self.source)


@dataclass(frozen=True)
class HarnessCandidate:
    """A complete runnable improvement, independent of whether it can be internalized."""
    candidate_id: str
    parent_revision: str
    patch: tuple[FileEdit, ...]
    full_revision: HarnessRevision
    rationale: str

    @classmethod
    def create(cls,store,parent,patch,rationale):
        if not rationale.strip(): raise ValueError("A main improvement rationale is required")
        full=store.apply(parent,patch)
        identity=digest([parent.version,full.version,rationale])
        return cls(identity,parent.version,tuple(patch),full,rationale)

    def validate(self,parent):
        if self.parent_revision!=parent.version or self.full_revision.policy!=parent.policy:
            raise ValueError("Candidate lineage/permissions mismatch")
        from ..harness.revision import text_hash
        files=parent.files();seen=set()
        for edit in self.patch:
            parent.policy.validate_path(edit.path)
            if edit.path in seen: raise ValueError("Repeated candidate patch path")
            seen.add(edit.path)
            if (text_hash(files[edit.path]) if edit.path in files else None)!=edit.before_hash:
                raise ValueError("Candidate patch base mismatch")
            if edit.content is None: files.pop(edit.path)
            else: files[edit.path]=edit.content
        if files!=self.full_revision.files(): raise ValueError("Candidate patch does not produce runnable full_revision")
        if self.candidate_id!=digest([parent.version,self.full_revision.version,self.rationale]):
            raise ValueError("Candidate identity mismatch")

    def to_dict(self):
        from dataclasses import asdict
        return {"candidate_id":self.candidate_id,"parent_revision":self.parent_revision,
            "patch":[asdict(p) for p in self.patch],"full_revision":self.full_revision.to_dict(),"rationale":self.rationale}

    @classmethod
    def from_dict(cls,value):
        return cls(value["candidate_id"],value["parent_revision"],tuple(FileEdit(**p) for p in value["patch"]),
                   HarnessRevision.from_dict(value["full_revision"]),value["rationale"])
