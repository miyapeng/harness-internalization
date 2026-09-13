"""Search identity contract; a fixed retrieval service remains an external prerequisite."""

def task_id(reset_payload):
    value = reset_payload.get("task_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Search-QA requires the dataset's actual task_id, never a placeholder")
    return value
