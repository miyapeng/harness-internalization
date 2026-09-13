from .module import HarnessModule

SOURCE = 'NAME = "review"\nKIND = "review"\nINSTRUCTION = "Review the unexecuted draft against the visible user constraints, available evidence and action syntax. Identify concrete violations and recommend a correction, without calling tools or inventing missing observations."\nPERSISTENCE = 0\n\ndef trigger(history, step):\n    return True\n'

def default_module():
    return HarnessModule.from_source(SOURCE)
