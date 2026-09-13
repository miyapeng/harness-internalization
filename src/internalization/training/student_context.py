from dataclasses import asdict
from ..core.types import State, Journal
from ..harness.module import Harness
from ..harness.runtime import TeacherHarness

class StudentHarness:
    def __init__(self, model, reduced: Harness, journal: Journal):
        self.runtime = TeacherHarness(model, reduced)
        self.journal = journal

    def augment(self, obs, task_ids, episode_ids, step, active):
        # Preserve the exact shared raw prompt in text_base. The target module
        # never enters this runtime; retained guidance is ephemeral prompt text.
        result = dict(obs)
        base = list(obs["text"])
        result["text_base"], result["text"] = base, list(base)
        for i, is_active in enumerate(active):
            if not is_active: continue
            state = State(str(task_ids[i]), str(episode_ids[i]), step, str(base[i]))
            guide = self.runtime.advise(state)
            result["text"][i] = guide.teacher_prompt
            self.journal.append("student_auxiliary_cost", state=asdict(state),
                student_prompt=guide.teacher_prompt, cost=asdict(guide.cost),
                active_modules=guide.active_modules, harness_version=guide.harness_version)
        return result

