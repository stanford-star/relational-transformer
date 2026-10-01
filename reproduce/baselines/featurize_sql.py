import json
import os
from pathlib import Path


def featurize_db(
    *,
    db: str,
    db_task_list: str,
    pre_dir: str,
    raw_dir: str,
    features_root: str,
    relbench_cache_dir: str,
) -> None:
    os.environ["RELBENCH_CACHE_DIR"] = str(Path(relbench_cache_dir).expanduser())
    import duckdb
    import numpy as np
    import pandas as pd
    from relbench.datasets import get_dataset

    from reproduce.baselines.rel2tab.featurizer import (
        get_table_splits,
        load_table_info,
        table_offset_and_len,
    )
    from reproduce.baselines.sql_queries import SQL_REGISTRY

    tables = sorted(
        {t for d, t in json.loads(Path(db_task_list).read_text()) if d == db}
    )
    assert tables, f"no tasks for {db} in {db_task_list}"

    con = duckdb.connect()
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET threads=4")
    rb_db = get_dataset(db, download=True).get_db(upto_test_timestamp=False)
    for tbl_name, tbl in rb_db.table_dict.items():
        con.register(tbl_name, tbl.df)
    print(f"[{db}] loaded {len(rb_db.table_dict)} tables into DuckDB", flush=True)

    out_dir = Path(features_root).expanduser() / db / "sql_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    for table in tables:
        entry = SQL_REGISTRY[db, table]
        sql, entity_col, time_col = entry["sql"], entry["entity_col"], entry["time_col"]

        min_offset, total_nodes = table_offset_and_len(pre_dir, db, table)
        splits_info = get_table_splits(load_table_info(pre_dir, db), table)
        ordered = sorted(splits_info.items(), key=lambda kv: kv[1]["node_idx_offset"])
        combined = pd.concat(
            [
                pd.read_parquet(
                    Path(raw_dir).expanduser() / db / "tasks" / table / f"{s}.parquet"
                ).reset_index(drop=True)
                for s, _ in ordered
            ],
            ignore_index=True,
        )
        assert len(combined) == total_nodes, (
            f"{db}/{table}: {len(combined)} task rows vs {total_nodes} nodes "
            f"in table_info.json -- the data the features are for is not the "
            f"data that was preprocessed"
        )

        key_cols = [time_col, entity_col]
        con.register("task_table", combined)
        feats_df = con.execute(sql).df()
        feats_df = feats_df.drop_duplicates(subset=key_cols, keep="first")
        merged = combined.merge(feats_df, on=key_cols, how="left")
        assert len(merged) == total_nodes, (
            f"{db}/{table}: merge changed the row count "
            f"({len(merged)} vs {total_nodes}); key columns are not unique"
        )
        feat_cols = [c for c in feats_df.columns if c not in key_cols]
        assert feat_cols, f"{db}/{table}: feature SQL produced no feature columns"

        arr = merged[feat_cols].values.astype(np.float32)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        np.nan_to_num(arr, copy=False, nan=0.0)

        mean = arr.mean(axis=0, keepdims=True)
        std = arr.std(axis=0, keepdims=True)
        std = np.where(std < 1e-8, 1.0, std)
        arr = (arr - mean) / std

        arr.astype(np.float32).tofile(out_dir / f"{table}_vectors.bin")
        (out_dir / f"{table}_meta.json").write_text(
            json.dumps(
                {
                    "n_features": arr.shape[1],
                    "min_offset": min_offset,
                    "total_nodes": total_nodes,
                }
            )
        )
        print(
            f"[{db}] {table}: {total_nodes} rows x {arr.shape[1]} features",
            flush=True,
        )
