"""Deterministic task scheduling, with a serializable cursor independent of model acceptance."""
import random
from .types import digest

BUDGET_SIZES = {"train":2048, "search":96, "dev":32,
                "retirement_0":128, "retirement_1":128, "retirement_2":128}


def validate_budget_manifest(manifest):
    manifest.validate_loop(3, versioned=True, cohort_minimum=30)
    for name, size in BUDGET_SIZES.items():
        if len(manifest.partition(name)) != size:
            raise ValueError(f"budget_v1 requires {name}={size}; no resizing or split borrowing")


def search_schedule(tasks, seed, cycles=3, count=8):
    if len(set(tasks)) != len(tasks) or len(tasks) < cycles*count:
        raise ValueError("Insufficient unique search tasks")
    order = sorted(tasks)
    random.Random(seed).shuffle(order)
    return tuple(tuple(order[i*count:(i+1)*count]) for i in range(cycles))


class TaskQueue:
    def __init__(self, tasks, seed, state=None):
        self.tasks = tuple(sorted(tasks))
        if len(set(tasks)) != len(tasks) or not tasks: raise ValueError("Unique training tasks required")
        self.seed, self.identity = seed, digest(self.tasks)
        self.epoch, self.position, self.draws = 0, 0, 0
        self.order = self._order(0)
        if state:
            if state["task_hash"] != self.identity or state["seed"] != seed:
                raise ValueError("Training sampler identity changed")
            self.epoch, self.position, self.draws = (state[k] for k in ("epoch","position","draws"))
            self.order = list(state["order"])
            if (sorted(self.order) != list(self.tasks) or any(type(n) is not int or n < 0 for n in
                    (self.epoch,self.position,self.draws)) or self.position > len(tasks)
                    or self.draws != self.epoch*len(tasks)+self.position):
                raise ValueError("Invalid persisted sampler cursor")

    def _order(self, epoch):
        order = list(self.tasks)
        random.Random(self.seed + epoch).shuffle(order)
        return order

    def take(self, count):
        if not 0 < count <= len(self.tasks): raise ValueError("Each batch needs distinct tasks")
        chosen = []
        while len(chosen) < count:
            if self.position == len(self.order):
                self.epoch += 1
                self.position = 0
                self.order = self._order(self.epoch)
                # Preserve every task while avoiding a duplicate across an epoch boundary.
                self.order = [t for t in self.order if t not in chosen]+[t for t in self.order if t in chosen]
            chosen.append(self.order[self.position])
            self.position += 1
            self.draws += 1
        return tuple(chosen)

    def state(self):
        return {"task_hash":self.identity,"seed":self.seed,"epoch":self.epoch,
                "position":self.position,"draws":self.draws,"order":list(self.order)}


def model_seed(base, batch, task, replica):
    return int(digest([base,batch,task,replica])[:8],16)


def seed_process(seed):
    import sys
    random.seed(seed)
    # Do not import GPU frameworks in proposer or environment processes just to seed them.
    if "numpy" in sys.modules: sys.modules["numpy"].random.seed(seed)
    if "torch" in sys.modules:
        torch = sys.modules["torch"]
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
