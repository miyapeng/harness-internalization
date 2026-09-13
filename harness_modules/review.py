NAME = "review"
KIND = "review"
INSTRUCTION = "Review the unexecuted draft against the visible user constraints, available evidence and action syntax. Identify concrete violations and recommend a correction, without calling tools or inventing missing observations."
PERSISTENCE = 0

def trigger(history, step):
    return True
