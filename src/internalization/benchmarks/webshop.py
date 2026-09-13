"""Preserved task identity/split contract. Full environment adapter is not run."""

def task_id(session):
    if type(session) is not int or session < 0: raise ValueError("Invalid WebShop session")
    return f"webshop:{session}"


def split_for(session):
    task_id(session)
    return "test" if session < 500 else "dev" if session < 1500 else "train"


def check_partition(sessions, partition):
    if any(split_for(session) != partition for session in sessions):
        raise ValueError("WebShop task is outside its official partition")
