import json
from pathlib import Path

import pandas as pd

from reproduce import config


def main() -> None:
    from relbench.datasets import get_dataset
    from relbench.tasks import get_task

    pairs = json.loads(
        (Path(config.pre_dir()) / "db-task-lists" / "forecast.json").read_text()
    )

    for db in sorted({d for d, _ in pairs}):
        rb_db = get_dataset(db, download=True).get_db(upto_test_timestamp=False)
        for tbl_name, tbl in rb_db.table_dict.items():
            raw_path = Path(config.raw_dir()) / db / "db" / f"{tbl_name}.parquet"
            assert raw_path.exists(), f"{db}: no raw parquet for table {tbl_name}"
            n_raw = pd.read_parquet(raw_path, columns=[]).shape[0]
            assert len(tbl.df) == n_raw, (
                f"{db}/{tbl_name}: classic db has {len(tbl.df)} rows, "
                f"raw parquet {n_raw}"
            )
        print(f"OK {db}: {len(rb_db.table_dict)} db tables match", flush=True)

    for db, task_name in pairs:
        rb_task = get_task(db, task_name, download=True)
        ent, ts = rb_task.entity_col, rb_task.time_col
        for split in ("train", "val", "test"):
            classic = rb_task.get_table(split).df.reset_index(drop=True)
            raw = pd.read_parquet(
                Path(config.raw_dir()) / db / "tasks" / task_name / f"{split}.parquet"
            ).reset_index(drop=True)
            assert len(classic) == len(raw), (
                f"{db}/{task_name}/{split}: {len(classic)} vs {len(raw)} rows"
            )
            assert list(classic.columns) == list(raw.columns), (
                f"{db}/{task_name}/{split}: columns differ"
            )
            key_c = list(zip(classic[ent].astype(str), classic[ts].astype(str)))
            key_r = list(zip(raw[ent].astype(str), raw[ts].astype(str)))
            assert len(set(key_c)) == len(classic), (
                f"{db}/{task_name}/{split}: (entity, time) keys not unique"
            )
            assert set(key_c) == set(key_r), (
                f"{db}/{task_name}/{split}: (entity, time) key sets differ"
            )
            label_cols = [c for c in classic.columns if c not in (ent, ts)]
            merged = classic.merge(raw, on=[ent, ts], suffixes=("_c", "_r"))
            assert len(merged) == len(classic)
            for col in label_cols:
                a = merged[f"{col}_c"].astype(float).fillna(-9e18)
                b = merged[f"{col}_r"].astype(float).fillna(-9e18)
                assert (a - b).abs().max() == 0.0, (
                    f"{db}/{task_name}/{split}: label {col!r} differs by key"
                )
            print(
                f"OK {db}/{task_name}/{split}: {len(classic)} rows, same keys "
                f"and labels",
                flush=True,
            )
    print("classic relbench and the raw collection agree", flush=True)


if __name__ == "__main__":
    main()
