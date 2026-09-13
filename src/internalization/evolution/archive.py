from dataclasses import asdict
import json

from ..core.types import Journal


class CandidateArchive:
    def __init__(self, path):
        self.path = path
        self.journal = Journal(path)

    def record(self, candidate, *, status, search_gain=None, dev_gain=None, reason=None):
        self.journal.append("candidate", **asdict(candidate), candidate_id=candidate.candidate_id,
            content_hash=candidate.content_hash, status=status, search_gain=search_gain,
            dev_gain=dev_gain, reason=reason)

    def history(self):
        if not self.path.exists(): return ()
        return tuple(json.loads(line) for line in self.path.read_text().splitlines())

    def record_attribution(self, candidate, *, result, checkpoint, h_plus, h_minus, evidence):
        self.journal.append("attribution", **asdict(candidate), candidate_id=candidate.candidate_id,
            content_hash=candidate.content_hash, status=result["decision"], reason=result["reason"],
            checkpoint=checkpoint, h_plus=h_plus.version, h_minus=h_minus.version,
            attribution=result, evidence=evidence)

    def proposer_history(self):
        # Search outcomes can inform proposals. Development and retirement
        # scores remain outside the proposer, preserving the original split rule.
        return tuple({key: row[key] for key in ("source", "parent", "version", "index",
            "candidate_id", "content_hash", "search_gain") if key in row}
            for row in self.history() if row["kind"] == "candidate")

    def record_outcome(self, candidate, *, result, evidence):
        # Full acceptance/retirement evidence is audit-only, like attribution.
        self.journal.append("outcome", **asdict(candidate), candidate_id=candidate.candidate_id,
            content_hash=candidate.content_hash, result=result, evidence=evidence)
