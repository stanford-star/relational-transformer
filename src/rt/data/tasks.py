import json
from dataclasses import dataclass
from pathlib import Path

from rt.data.resolve import read_meta

_TASK_TYPE = {"binary_classification": "clf", "regression": "reg"}


@dataclass(frozen=True)
class Task:
    db_name: str
    table_name: str
    target_column: str
    task_type: str
    split: str = ""
    leakage_columns: tuple[tuple[str, str], ...] = ()


def resolve_db_task_list(db_task_list) -> list[tuple[str, str]]:
    if isinstance(db_task_list, str):
        p = Path(db_task_list).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"db_task_list {db_task_list!r} does not exist. The released "
                f"lists ship with the preprocessed data, as "
                f"<pre_dir>/db-task-lists/<name>.json"
            )
        pairs = json.loads(p.read_text())
    else:
        pairs = db_task_list
    out = []
    for pair in pairs:
        db, name = pair
        out.append((str(db), str(name)))
    return out


def get_tasks(pre_dir, db_task_list, splits) -> list[Task]:
    pairs = resolve_db_task_list(db_task_list)
    by_db: dict[str, list[str]] = {}
    for db, name in pairs:
        by_db.setdefault(db, []).append(name)

    out: list[Task] = []
    ignored: list[str] = []
    for db, names in by_db.items():
        meta = read_meta(pre_dir, db)
        explicit = {
            t["name"]: t
            for t in meta.get("tasks", [])
            if _TASK_TYPE.get(t.get("task_type")) and t.get("target_col")
        }
        for name in names:
            t = explicit.get(name)
            if t is None:
                ignored.append(f"{db}/{name}")
                continue
            tt = _TASK_TYPE[t["task_type"]]
            leaks = tuple(
                (str(tbl), str(col)) for tbl, col in t.get("remove_columns") or ()
            )
            if t.get("kind") != "autocomplete" and not t.get("splits"):
                raise ValueError(
                    f"{db}: task {name!r} has kind={t.get('kind')!r} and no splits, "
                    f"so it resolves to no tasks; its meta.json entry is stale "
                    f"(re-preprocess the db)"
                )
            if t.get("kind") == "autocomplete":
                if "train" in splits:
                    out.append(
                        Task(db, t["entity_table"], t["target_col"], tt, "train", leaks)
                    )
                continue
            for split in splits:
                if split in t.get("splits", []):
                    out.append(Task(db, name, t["target_col"], tt, split, leaks))

    if ignored:
        print(f"ignored {len(ignored)} task(s) this build cannot predict:", flush=True)
        for name in ignored:
            print(f"  {name}", flush=True)
    return out
