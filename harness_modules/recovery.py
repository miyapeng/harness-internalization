NAME = "recovery"
KIND = "recovery"
INSTRUCTION = "Diagnose repeated failure using only visible history. Identify the unmet precondition or missing relation. Recommend a different next action or query; do not repeat failed actions or assume unobserved tool results."
PERSISTENCE = 2

def trigger(history, step):
    return "nothing happens" in history.lower() or "invalid" in history.lower() or "no results" in history.lower()
