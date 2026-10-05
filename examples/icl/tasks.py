from examples.icl import env


def task_list() -> list[tuple[str, str]]:
    from rt.data import resolve_db_task_list

    pairs = resolve_db_task_list(env.env("RT_TASKS"))
    assert pairs, f"{env.env('RT_TASKS')} lists no (db, task) pairs"
    return pairs
