"""Explicit API regression adapter. Production uses ClaudeCodeProposer."""
from dataclasses import asdict
from .proposer import APITransport
from .materialization import (SPEC_PATH, materialize_candidates, public_execution_trace,
    search_only_history, validate_evidence, validate_search_input)

CONTRACT = SPEC_PATH.read_text() + "\nTest API transport only: replace workspace with patch (path/content edits); return candidates JSON."


class CodeProposer(APITransport):
    """Explicit test/reference API adapter; never selected by production entrypoints."""
    def __init__(self,store,**kwargs):
        super().__init__(**kwargs)
        self.store=store

    def propose(self,request):
        validate_search_input(request)
        public={"candidate_count":request.count,"parent_revision":request.harness.version,
            "files":request.harness.files(),"workspace_policy":asdict(request.harness.policy),
            "traces":[public_execution_trace(t) for t in request.trajectories],
            "scores":[asdict(s) for s in request.scores],"history":search_only_history(request.history)}
        raw=self.request_json(CONTRACT,public,request.output)
        if not isinstance(raw,dict) or set(raw)!={'candidates'} or not isinstance(raw['candidates'],list):
            raise ValueError("Expected exactly the candidates JSON field")
        proposals=raw['candidates']
        if len(proposals)!=request.count: raise ValueError("Wrong code candidate count")
        return materialize_candidates(self.store,request,proposals)
