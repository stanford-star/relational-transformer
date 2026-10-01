import json
from pathlib import Path

from reproduce import config


def tasks() -> list[tuple[str, str]]:
    path = Path(config.pre_dir()) / "db-task-lists" / "forecast.json"
    pairs = [tuple(p) for p in json.loads(path.read_text())]
    assert len(pairs) == 21, f"{path}: {len(pairs)} tasks, the paper's grid has 21"
    return pairs


def reg_tasks() -> list[tuple[str, str]]:
    from rt.data import get_tasks

    return [
        (t.db_name, t.table_name)
        for t in get_tasks(config.pre_dir(), config.db_task_list(), ("test",))
        if t.task_type == "reg"
    ]
