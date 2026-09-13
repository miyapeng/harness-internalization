from .module import HarnessModule

SOURCE = 'NAME = "recovery"\nKIND = "recovery"\nINSTRUCTION = "Diagnose repeated failure using only visible history. Identify the unmet precondition or missing relation. Recommend a different next action or query; do not repeat failed actions or assume unobserved tool results."\nPERSISTENCE = 2\n\ndef trigger(history, step):\n    return "nothing happens" in history.lower() or "invalid" in history.lower() or "no results" in history.lower()\n'

def default_module():
    return HarnessModule.from_source(SOURCE)
