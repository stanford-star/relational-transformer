from reproduce import config
from rt.data import get_mixture


def tasks() -> list[tuple[str, str]]:
    pairs = get_mixture("relbench", "forecast")
    assert len(pairs) == 21, f"{len(pairs)} tasks, the paper's grid has 21"
    return pairs


def reg_tasks() -> list[tuple[str, str]]:
    from rt.data import get_tasks

    return [
        (t.db_name, t.table_name)
        for t in get_tasks(config.pre_dir(), config.db_task_list(), ("test",))
        if t.task_type == "reg"
    ]
