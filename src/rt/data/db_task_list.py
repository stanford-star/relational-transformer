import json
from pathlib import Path

import yaml

from rt.data.resolve import read_meta

SUPPORTED_TASK_TYPES = ("binary_classification", "regression")
KINDS = ("forecast", "autocomplete")


def _databases(pre_dir: str) -> list[str]:
    p = Path(pre_dir).expanduser()
    return sorted(d.name for d in p.iterdir() if (d / "meta.json").is_file())


def _has_target(task: dict, column_index: dict) -> bool:
    target = task["target_col"]
    return any(
        f"{target} of {table}" in column_index
        for table in (task["entity_table"], task["name"])
    )


def _kind(task: dict) -> str:
    return "autocomplete" if task.get("kind") == "autocomplete" else "forecast"


def make_db_task_list(pre_dir: str, kinds=KINDS) -> list[tuple[str, str]]:
    p = Path(pre_dir).expanduser()
    out = []
    for db in _databases(pre_dir):
        column_index = json.loads((p / db / "column_index.json").read_text())
        for task in read_meta(pre_dir, db).get("tasks", []):
            if task.get("task_type") not in SUPPORTED_TASK_TYPES:
                continue
            if _kind(task) not in kinds:
                continue
            if not _has_target(task, column_index):
                continue
            out.append((db, task["name"]))
    return out


def _keep_plurel_column(
    stats: dict | None, task_type: str, max_majority_frac: float, min_std: float
) -> bool:
    if stats is None:
        return True
    if stats.get("is_source_node", False):
        return False
    if stats.get("n_unique", 0) < 2:
        return False
    if task_type == "binary_classification":
        return (
            stats.get("n_classes", 0) >= 2
            and stats.get("majority_frac", 0.0) <= max_majority_frac
        )
    return stats.get("std", 0.0) >= min_std


def make_plurel_db_task_list(
    pre_dir: str,
    raw_dir: str,
    num_dbs: int = 1900,
    max_fkeys: int = 5,
    max_majority_frac: float = 0.99,
    min_std: float = 1e-4,
) -> list[tuple[str, str]]:
    raw = Path(raw_dir).expanduser()
    dbs = sorted(_databases(pre_dir), key=lambda d: int(d.rsplit("-", 1)[1]))
    out, kept = [], 0
    for db in dbs:
        manifest = yaml.safe_load((raw / db / "manifest.yaml").read_text())
        if any(
            len(t.get("fkeys") or {}) > max_fkeys for t in manifest["tables"].values()
        ):
            continue
        scores = json.loads((raw / db / "scores.json").read_text())
        for task in read_meta(pre_dir, db).get("tasks", []):
            if task.get("task_type") not in SUPPORTED_TASK_TYPES:
                continue
            stats = scores.get(task["entity_table"], {}).get(task["target_col"])
            if _keep_plurel_column(
                stats, task["task_type"], max_majority_frac, min_std
            ):
                out.append((db, task["name"]))
        kept += 1
        if kept == num_dbs:
            break
    assert kept == num_dbs, f"only {kept} of {num_dbs} databases pass the filter"
    return sorted(out)


def write_db_task_list(pairs, path: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([list(pair) for pair in pairs], indent=1) + "\n")
    print(f"{len(pairs)} tasks over {len({db for db, _ in pairs})} dbs -> {p}")
    return str(p)
