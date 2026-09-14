"""Search/dev rules unchanged; each candidate's complete code tree is actually run."""
from dataclasses import asdict, dataclass
from .candidate import HarnessCandidate
from ..harness.revision import HarnessRevision
from ..core.types import EpisodeResult
from .search import evaluate_tasks
from ..evaluation.retirement import paired_interval
from ..core.types import write_json



def dev_acceptance_policy(policy, *, budget=False):
    """Record exactly the existing search/dev gate, not the attribution gate."""
    return {"decision_source":"dev", "rule":("search paired_mean > 0; top one only; dev_gain.low > 0" if budget else "search_gain.low > 0 and dev_gain.low > 0"),
        "confidence":policy.confidence,"bootstrap_samples":policy.bootstrap_samples,
        "seed":policy.seed,"bootstrap_cluster":"task_id","minimum_lower_gain":0}


@dataclass(frozen=True)
class RevisionSelection:
    candidate: HarnessCandidate
    parent: HarnessRevision
    checkpoint: str
    parent_dev: tuple[EpisodeResult, ...]
    candidate_dev: tuple[EpisodeResult, ...]
    search_gain: dict
    dev_gain: dict
    dev_accepted: bool
    search_accepted: bool
    index: int
    policy: dict
    model_version: str | None = None

    @property
    def accepted(self):
        return self.search_accepted and self.dev_accepted

    def acceptance_record(self):
        return {"decision_source":"dev", "passed":self.accepted,
            "dev_accepted":self.dev_accepted,"search_accepted":self.search_accepted,
            "checkpoint":self.checkpoint,"model_version":self.model_version,
            "candidate_id":self.candidate.candidate_id,"parent_revision":self.parent.version,
            "full_revision":self.candidate.full_revision.version,
            "task_ids":sorted({row.task_id for row in self.candidate_dev}),
            "seeds":sorted({row.seed for row in self.candidate_dev}),
            "parent_dev_results":[asdict(row) for row in self.parent_dev],
            "candidate_dev_results":[asdict(row) for row in self.candidate_dev],
            "search_gain":self.search_gain,"dev_gain":self.dev_gain,"policy":self.policy,
            "evidence":{"parent":"baseline_dev/episodes.json",
                "candidate":f"candidate_{self.index}/dev/episodes.json"},
            "reused_evaluations":True,"additional_evaluation_calls":0}


def search_revisions(components,request,*,baseline_search,baseline_dev,dev_tasks,seeds,policy,journal):
    if components.execution_config and components.execution_config["schedule"]["profile"] == "budget_v1":
        return budget_search_revisions(components,request,baseline_search=baseline_search,baseline_dev=baseline_dev,
            dev_tasks=dev_tasks,seeds=seeds,policy=policy,journal=journal)
    candidates=components.proposer.propose(request)
    if len(candidates)!=request.count: raise ValueError("Proposer returned wrong code candidate count")
    evaluated=[]
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
            versions={t.model_version for result in (baseline_search,baseline_dev,search,dev) for t in result.trajectories}
            if len(versions)>1: raise ValueError("Search/dev evaluations used different model snapshots")
            selection=RevisionSelection(candidate,request.harness,request.checkpoint,
                tuple(baseline_dev.evaluations),tuple(dev.evaluations),sg,dg,dg["low"]>0,sg["low"]>0,
                index,dev_acceptance_policy(policy),next(iter(versions),None))
            useful=selection.accepted
            write_json(folder/"dev_acceptance.json",selection.acceptance_record())
            journal.append("code_candidate",candidate=candidate.to_dict(),search_gain=sg,dev_gain=dg,
                           status="eligible" if useful else "no_gain")
            evaluated.append(selection)
        except Exception as exc:
            write_json(folder/"rejection.json",{"reason":str(exc),"index":index})
            journal.append("candidate_failed",index=index,reason=str(exc),
                           candidate=candidate.to_dict() if isinstance(candidate,HarnessCandidate) else candidate)
            # A failing candidate cannot abort evaluation of its sibling or mutate its parent.
        request.harness.files()
    # A rejected best candidate can be returned for audit, but never as an accepted improvement.
    eligible=[item for item in evaluated if item.accepted]
    pool=eligible or evaluated
    return max(pool,key=lambda item:(item.dev_gain["mean"],
        -sum(row.cost.total_tokens for row in item.candidate_dev))) if pool else None


def public_history(journal):
    import json
    if not journal.path.exists(): return ()
    rows=(json.loads(line) for line in journal.path.read_text().splitlines())
    return tuple(({"candidate":row["candidate"],"search_gain":row["search_gain"],"status":row["status"]}
        if row["kind"]=="code_candidate" else {"candidate":row["candidate"],"status":"failed",
            "reason":"candidate_execution_or_evaluation_failed"})
        for row in rows if row["kind"] in ("code_candidate","candidate_failed"))


def budget_search_revisions(components, request, *, baseline_search, baseline_dev, dev_tasks, seeds, policy, journal):
    """Two common-task evaluations; positive paired mean only; a single dev finalist."""
    candidates = components.proposer.propose(request)
    if len(candidates) != request.count: raise ValueError("Wrong candidate count")
    eligible = []
    for index, candidate in enumerate(candidates):
        folder = request.output.parent/f"candidate_{index}"
        try:
            if not isinstance(candidate, HarnessCandidate): raise ValueError(str(candidate))
            candidate.validate(request.harness)
            write_json(folder/"candidate.json", candidate.to_dict())
            result = evaluate_tasks(components.runner, request.checkpoint, candidate.full_revision,
                                    request.tasks, seeds, folder/"search")
            parent_rows = {(r.task_id,r.seed):r for r in baseline_search.evaluations}
            rows = {(r.task_id,r.seed):r for r in result.evaluations}
            if len(rows) != len(result.evaluations) or rows.keys() != parent_rows.keys():
                raise ValueError("Unpaired search results")
            versions = {t.model_version for r in (baseline_search,result) for t in r.trajectories}
            if len(versions)>1: raise ValueError("Search used different model snapshots")
            gains = [rows[key].success-parent_rows[key].success for key in sorted(rows)]
            gain = {"mean":sum(gains)/len(gains), "paired_gains":gains,
                    "rule":"paired_mean > 0", "significance_test":False,
                    "task_ids":list(request.tasks)}
            journal.append("code_candidate", candidate=candidate.to_dict(), search_gain=gain,
                           status="search_positive" if gain["mean"]>0 else "no_gain")
            write_json(folder/"search_screen.json",gain)
            if gain["mean"]>0: eligible.append((gain["mean"], -sum(r.cost.total_tokens for r in result.evaluations),
                                               -index, candidate, gain, versions))
        except Exception as exc:
            write_json(folder/"rejection.json", {"reason":str(exc),"index":index})
            journal.append("candidate_failed",index=index,reason=str(exc),
                           candidate=candidate.to_dict() if isinstance(candidate,HarnessCandidate) else candidate)
        request.harness.files()
    if not eligible: return None
    _, _, neg_index, candidate, gain, versions = max(eligible, key=lambda row:row[:3])
    index = -neg_index
    folder = request.output.parent/f"candidate_{index}"
    try:
        if baseline_dev is None:
            baseline_dev = evaluate_tasks(components.runner,request.checkpoint,request.harness,dev_tasks,seeds,
                                          request.output.parent/"baseline_dev")
        dev = evaluate_tasks(components.runner,request.checkpoint,candidate.full_revision,dev_tasks,seeds,folder/"dev")
        versions |= {t.model_version for result in (baseline_dev,dev) for t in result.trajectories}
        if len(versions)>1: raise ValueError("Search/dev used different model snapshots")
        dg = paired_interval(dev.evaluations,baseline_dev.evaluations,lambda r:r.success,policy)
        selected = RevisionSelection(candidate,request.harness,request.checkpoint,
            tuple(baseline_dev.evaluations),tuple(dev.evaluations),gain,dg,dg["low"]>0,True,index,
            dev_acceptance_policy(policy, budget=True),next(iter(versions),None))
        write_json(folder/"dev_acceptance.json",selected.acceptance_record())
        return selected
    except Exception as exc:
        write_json(folder/"dev_rejection.json",{"reason":str(exc),"no_second_finalist":True})
        return None
