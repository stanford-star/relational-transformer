import json
from pathlib import Path

import numpy as np

from reproduce import config

ARMS: list[tuple[str, str, str, str | None]] = [
    ("fulltest", "rt-j", "fulltest/rt", None),
    (
        "fulltest",
        "rdblearn_tabicl",
        "fulltest/rdblearn_tabicl",
        "fulltest_ext/rdblearn_tabicl",
    ),
    ("fulltest", "sql_tabicl", "fulltest/sql_tabicl", "fulltest_ext/sql_tabicl"),
    ("fulltest", "rdblearn_lgbm", "fulltest/rdblearn_lgbm", None),
    ("fulltest", "sql_lgbm", "fulltest/sql_lgbm", None),
    ("subsampled", "rt-j", "subsampled/rt", None),
    ("subsampled", "rdblearn_tabicl", "subsampled/rdblearn_tabicl", None),
    ("subsampled", "sql_tabicl", "subsampled/sql_tabicl", None),
    ("subsampled", "rdblearn_lgbm", "subsampled/rdblearn_lgbm", None),
    ("subsampled", "sql_lgbm", "subsampled/sql_lgbm", None),
    ("abl", "rw", "subsampled/rt", None),
    ("abl", "sem", "subsampled/rt", None),
    ("abl", "nosem", "abl/nosem", None),
    ("abl", "rand", "abl/rand", None),
    ("abl", "bfs32", "abl/bfs32", None),
    ("abl", "bfs256", "abl/bfs256", None),
    ("abl", "vdb_rdblearn", "abl/vdb_rdblearn", None),
    ("abl", "vdb_rt", "abl/vdb_rt", None),
]

N_TASKS = 21
N_EXT_TASKS = 9


def load_arm(arm_dir: Path, ext_dir: Path | None) -> list[dict] | None:
    paths = sorted(arm_dir.glob("*__*.json"))
    if len(paths) != N_TASKS:
        print(f"{arm_dir}: {len(paths)}/{N_TASKS} task JSONs, not reduced", flush=True)
        return None
    recs = [json.loads(p.read_text()) for p in paths]
    if ext_dir is None:
        return recs
    ext: dict[str, dict] = {}
    for p in sorted(ext_dir.glob("**/*__*.json")):
        rec = json.loads(p.read_text())
        per_ctx = ext.setdefault(rec["task"], {})
        assert not set(per_ctx) & set(rec["per_ctx"]), f"{p}: duplicate context size"
        per_ctx.update(rec["per_ctx"])
    n_ctx = max((len(v) for v in ext.values()), default=0)
    complete = [t for t, v in ext.items() if len(v) == n_ctx]
    if len(complete) != N_EXT_TASKS or n_ctx == 0:
        print(
            f"{ext_dir}: {len(complete)}/{N_EXT_TASKS} tasks have all {n_ctx} "
            f"extension context sizes, not reduced",
            flush=True,
        )
        return None
    for r in recs:
        if r["task"] in ext:
            assert not set(r["per_ctx"]) & set(ext[r["task"]])
            r["per_ctx"].update(ext[r["task"]])
    return recs


def series(recs: list[dict]) -> dict:
    ctxs = sorted({int(c) for r in recs for c in r["per_ctx"]})
    rows = []
    for ctx in ctxs:
        row: dict = {"ctx_size": ctx, "per_task": {}}
        by_type = {"clf": [], "reg": []}
        for r in recs:
            if str(ctx) not in r["per_ctx"]:
                continue
            entry = r["per_ctx"][str(ctx)]
            row["per_task"][r["task"]] = {
                "metric_name": entry["metric_name"],
                "metric_value": entry["metric_value"],
                "mean_labels": entry["mean_labels"],
            }
            by_type[r["task_type"]].append(entry["metric_value"])
        for tt, key in (("clf", "avg_auc"), ("reg", "avg_mae")):
            n_type = sum(r["task_type"] == tt for r in recs)
            if len(by_type[tt]) == n_type:
                row[key] = float(np.mean(by_type[tt]))
            else:
                assert not by_type[tt], (
                    f"ctx={ctx}: {len(by_type[tt])}/{n_type} {tt} tasks scored"
                )
        rows.append(row)
    return {
        "n_tasks": len(recs),
        "task_types": {r["task"]: r["task_type"] for r in recs},
        "ctx_sizes": ctxs,
        "rows": rows,
    }


def main() -> None:
    root = Path(config.out_root()) / "scaling"
    dest_root = config.series_dir() / "scaling"
    done = 0
    for group, name, arm, ext in ARMS:
        recs = load_arm(root / arm, root / ext if ext else None)
        if recs is None:
            continue
        out = series(recs) | {"group": group, "arm": name, "arm_dir": arm}
        dest = dest_root / group / f"{name}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        last = out["rows"][-1]
        print(
            f"wrote {dest} ({len(out['rows'])} context sizes; at "
            f"ctx={last['ctx_size']} auc={last.get('avg_auc', float('nan')):.4f} "
            f"mae={last.get('avg_mae', float('nan')):.4f})",
            flush=True,
        )
        done += 1
    print(f"{done}/{len(ARMS)} arms reduced", flush=True)


if __name__ == "__main__":
    main()
