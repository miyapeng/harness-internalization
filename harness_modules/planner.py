NAME = "planner"
KIND = "planner"
INSTRUCTION = "Plan the next subgoal from the visible goal and history. Check action preconditions and missing evidence. Give concise guidance for the next action, without executing tools or claiming unseen facts."
PERSISTENCE = 1

def trigger(history, step):
    return step == 0 or "nothing happens" in history.lower()
