"""HotpotQA distractor-context agent wrapper, with explicit retrieval and final answer.

Metric equations independently expressed from the published HotpotQA protocol.
No upstream source or dataset is bundled. See THIRD_PARTY_NOTICES.md.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import json
import re
import string

from .common import Catalog, action_json
from ..core.types import write_json
from ..training.rollout import EnvironmentStep

STATUS = "implemented_cpu_fixture_verified_official_data_not_run"


def normalize(text):
    words = re.sub(r"\b(?:a|an|the)\b", " ", text.lower().translate(str.maketrans("", "", string.punctuation)))
    return " ".join(words.split())


def prf(overlap, predicted, gold):
    p, r = overlap/predicted if predicted else 0., overlap/gold if gold else 0.
    return p, r, 2*p*r/(p+r) if p+r else 0.


def metrics(answer, facts, gold):
    left, right = normalize(answer), normalize(gold["answer"])
    shared = sum((Counter(left.split()) & Counter(right.split())).values())
    if left != right and {left, right} & {"yes", "no", "noanswer"}: shared = 0
    p, r, f = prf(shared, len(left.split()), len(right.split()))
    predicted, expected = set(map(tuple, facts)), set(map(tuple, gold["supporting_facts"]))
    sp, sr, sf = prf(len(predicted & expected), len(predicted), len(expected))
    jp, jr = p*sp, r*sr
    return {"em":float(left == right), "prec":p, "recall":r, "f1":f,
        "sp_em":float(predicted == expected), "sp_prec":sp, "sp_recall":sr, "sp_f1":sf,
        "joint_em":float(left == right and predicted == expected), "joint_prec":jp,
        "joint_recall":jr, "joint_f1":2*jp*jr/(jp+jr) if jp+jr else 0.}


class HotpotQAEnvironment:
    def __init__(self, config, output, *, training=False):
        if config.options.get("setting", "distractor") != "distractor":
            raise ValueError("This adapter implements distractor context, not fullwiki retrieval")
        self.config, self.output, self.training = config, output, training
        self.catalog = Catalog(config.catalog, "hotpotqa")
        self.done = True

    def reset(self, task_id, seed):
        row = self.catalog.get(task_id, self.training)
        self.record = row["record"]
        if "answer" not in self.record or "supporting_facts" not in self.record:
            raise ValueError("Unlabelled HotpotQA test requires server evaluation; no local reward is available")
        self.task_id, self.seed, self.steps, self.done = task_id, seed, 0, False
        self.docs = dict(self.record["context"])
        if len(self.docs) != len(self.record["context"]): raise ValueError("Duplicate context titles")
        self.history = [self.record["question"],
            'Use JSON actions: {"action":"search","query":"..."}, {"action":"lookup","title":"..."}, '
            '{"action":"final","answer":"...","supporting_facts":[["Title",0]]}. '
            'Search is restricted to the supplied distractor paragraphs. Sentence indices start at zero.',
            "Available titles: " + json.dumps(list(self.docs), ensure_ascii=False)]
        write_json(self.output/"identity.json", {"task_id":task_id, "seed":seed,
            "catalog_hash":self.catalog.fingerprint, "setting":"distractor"})
        return {"observation":"\n".join(self.history)}

    def step(self, action):
        if self.done: raise RuntimeError("Episode is already finished")
        self.steps += 1
        valid, calls, answer, facts = True, 0, "", []
        try:
            item = action_json(action)
            kind = item["action"]
            if kind == "search":
                query = item["query"]
                if not isinstance(query, str) or not query.strip(): raise ValueError("Nonempty query required")
                terms = set(normalize(query).split())
                ranked = sorted(self.docs, key=lambda title: (-len(terms & set(normalize(title+" "+" ".join(self.docs[title])).split())), title))
                count = int(self.config.options.get("top_k", 3))
                if count < 1: raise ValueError("Positive top_k required")
                obs = json.dumps({t:list(enumerate(self.docs[t])) for t in ranked[:count]}, ensure_ascii=False)
                calls = 1
            elif kind == "lookup":
                obs = json.dumps(list(enumerate(self.docs[item["title"]])), ensure_ascii=False)
                calls = 1
            elif kind == "final":
                answer, facts = item["answer"], item["supporting_facts"]
                if not isinstance(answer, str) or not isinstance(facts, list): raise ValueError("Invalid final answer")
                if any(not isinstance(f, list) or len(f)!=2 or not isinstance(f[0], str) or type(f[1]) is not int for f in facts):
                    raise ValueError("Supporting facts must be [title, sentence_index] pairs")
                self.done, obs = True, "Answer submitted."
            else: raise ValueError("Unknown action")
        except (KeyError, TypeError, ValueError) as exc:
            valid, obs, answer, facts = False, "Invalid action: "+str(exc), "", []
        self.done = self.done or self.steps >= self.config.max_steps
        reward = 0.
        if self.done:
            values = metrics(answer, facts, self.record)
            reward = values["joint_f1"]
            write_json(self.output/"grade.json", {"task_id":self.task_id, "seed":self.seed,
                "prediction":{"answer":answer, "sp":facts}, "metrics":values, "reward_metric":"joint_f1"})
        self.history.extend(["Action: "+action, "Observation: "+obs])
        return asdict(EnvironmentStep("\n".join(self.history), reward, self.done, reward, valid, calls))

    def close(self): pass
