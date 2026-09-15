from dataclasses import dataclass, replace
from ..core.types import digest
from ..harness.revision import HarnessRevision, FileEdit, InternalizationTarget


@dataclass(frozen=True)
class HarnessCandidate:
    """A complete runnable improvement, independent of whether it can be internalized."""
    candidate_id: str
    parent_revision: str
    patch: tuple[FileEdit, ...]
    full_revision: HarnessRevision
    rationale: str
    evidence_refs: tuple[dict, ...] = ()
    internalization_target: InternalizationTarget | None = None

    def identity(self):
        target=self.internalization_target
        boundary=None if target is None else [target.full_revision.version,target.reduced_revision.version,
            target.removed_behavior,target.supervision_adapter,target.target_control_id]
        return digest([self.parent_revision,self.full_revision.version,self.rationale,self.evidence_refs,boundary])

    def with_internalization(self,target):
        if target is not None:
            if not isinstance(target,InternalizationTarget) or target.full_revision.version!=self.full_revision.version:
                raise ValueError("Embedded target must belong to the candidate full revision")
            target.validate_structure()
        result=replace(self,internalization_target=target)
        return replace(result,candidate_id=result.identity())

    @classmethod
    def create(cls,store,parent,patch,rationale,*,evidence_refs=()):
        if not isinstance(rationale,str) or not rationale.strip(): raise ValueError("A main improvement rationale is required")
        full=store.apply(parent,patch)
        result=cls('',parent.version,tuple(patch),full,rationale,tuple(dict(ref) for ref in evidence_refs))
        return replace(result,candidate_id=result.identity())

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
        if self.internalization_target is not None:
            self.with_internalization(self.internalization_target)
        if self.candidate_id!=self.identity():
            raise ValueError("Candidate identity mismatch")

    def to_dict(self):
        from dataclasses import asdict
        return {"candidate_id":self.candidate_id,"parent_revision":self.parent_revision,
            "patch":[asdict(p) for p in self.patch],"full_revision":self.full_revision.to_dict(),"rationale":self.rationale,
            "evidence_refs":list(self.evidence_refs),
            "internalization_target":self.internalization_target.to_dict() if self.internalization_target else None}

    @classmethod
    def from_dict(cls,value):
        return cls(value["candidate_id"],value["parent_revision"],tuple(FileEdit(**p) for p in value["patch"]),
                   HarnessRevision.from_dict(value["full_revision"]),value["rationale"],tuple(value["evidence_refs"]),
                   InternalizationTarget.from_dict(value["internalization_target"]) if value["internalization_target"] is not None else None)
