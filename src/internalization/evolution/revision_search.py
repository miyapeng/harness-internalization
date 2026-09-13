"""Search/dev rules unchanged; each candidate's complete code tree is actually run."""
from .candidate import HarnessCandidate
from .search import evaluate_tasks
from ..evaluation.retirement import paired_interval
from ..core.types import write_json


def search_revisions(components,request,*,baseline_search,baseline_dev,dev_tasks,seeds,policy,journal):
    candidates=components.proposer.propose(request)
    if len(candidates)!=request.count: raise ValueError("Proposer returned wrong code candidate count")
    eligible=[]
    for index,candidate in enumerate(candidates):
        folder=request.output.parent/f"candidate_{index}"
        try:
            if not isinstance(candidate,HarnessCandidate): raise ValueError(str(candidate))
            candidate.validate(request.harness)
            write_json(folder/"candidate.json",candidate.to_dict())
            search=evaluate_tasks(components.runner,request.checkpoint,candidate.full_revision,request.tasks,seeds,folder/"search")
            dev=evaluate_tasks(components.runner,request.checkpoint,candidate.full_revision,dev_tasks,seeds,folder/"dev")
            sg=paired_interval(search.evaluations,baseline_search.evaluations,lambda r:r.success,policy)
            dg=paired_interval(dev.evaluations,baseline_dev.evaluations,lambda r:r.success,policy)
            useful=sg["low"]>0 and dg["low"]>0
            journal.append("code_candidate",candidate=candidate.to_dict(),search_gain=sg,dev_gain=dg,
                           status="eligible" if useful else "no_gain")
            if useful: eligible.append((dg["mean"],-sum(r.cost.total_tokens for r in dev.evaluations),candidate))
        except Exception as exc:
            write_json(folder/"rejection.json",{"reason":str(exc),"index":index})
            journal.append("candidate_failed",index=index,reason=str(exc),
                           candidate=candidate.to_dict() if isinstance(candidate,HarnessCandidate) else candidate)
            # A failing candidate cannot abort evaluation of its sibling or mutate its parent.
        request.harness.files()
    return max(eligible,key=lambda item:item[:2])[2] if eligible else None


def public_history(journal):
    import json
    if not journal.path.exists(): return ()
    rows=(json.loads(line) for line in journal.path.read_text().splitlines())
    return tuple(({"candidate":row["candidate"],"search_gain":row["search_gain"],"status":row["status"]}
        if row["kind"]=="code_candidate" else {"candidate":row["candidate"],"status":"failed",
            "reason":"candidate_execution_or_evaluation_failed"})
        for row in rows if row["kind"] in ("code_candidate","candidate_failed"))
