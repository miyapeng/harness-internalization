from .module import HarnessModule

SOURCE = 'NAME = "planner"\nKIND = "planner"\nINSTRUCTION = "Plan the next subgoal from the visible goal and history. Check action preconditions and missing evidence. Give concise guidance for the next action, without executing tools or claiming unseen facts."\nPERSISTENCE = 1\n\ndef trigger(history, step):\n    return step == 0 or "nothing happens" in history.lower()\n'

def default_module():
    return HarnessModule.from_source(SOURCE)
