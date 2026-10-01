import json
import os
import time
from pathlib import Path


def rdb_dataset(db: str):
    import relbench.base
    from rdblearn.datasets import RDBDataset

    orig_get_db = relbench.base.Dataset.get_db

    def full_get_db(self, *a, **kw):
        return orig_get_db(self, upto_test_timestamp=False)

    full_get_db.cache_clear = lambda: None
    relbench.base.Dataset.get_db = full_get_db
    try:
        return RDBDataset.from_relbench(db)
    finally:
        relbench.base.Dataset.get_db = orig_get_db


def featurize_table(
    *,
    db: str,
    table: str,
    task_type: str,
    pre_dir: str,
    raw_dir: str,
    features_root: str,
    relbench_cache_dir: str,
    max_depth: int,
    max_train_samples: int,
) -> None:
    os.environ["RELBENCH_CACHE_DIR"] = str(Path(relbench_cache_dir).expanduser())
    import fastdfs
    import numpy as np
    import pandas as pd
    from fastdfs import DFSConfig
    from rdblearn.config import RDBLearnConfig
    from rdblearn.estimator import RDBLearnEstimator
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import make_pipeline

    from reproduce.baselines.rel2tab.featurizer import (
        get_table_splits,
        load_table_info,
        table_offset_and_len,
    )

    out_dir = Path(features_root).expanduser() / db / "rdblearn_features"
    out_dir.mkdir(parents=True, exist_ok=True)
    vectors_path = out_dir / f"{table}_vectors.bin"
    meta_path = out_dir / f"{table}_meta.json"
    if vectors_path.exists() and meta_path.exists():
        print(f"[{db}] {table}: already featurized, skipping", flush=True)
        return

    config = RDBLearnConfig(
        dfs=DFSConfig(
            max_depth=max_depth,
            agg_primitives=["max", "min", "mean", "count", "mode", "std"],
            engine="dfs2sql",
        ),
        enable_target_augmentation=False,
        max_train_samples=max_train_samples,
        predict_batch_size=5000,
    )

    tic = time.time()
    dataset = rdb_dataset(db)

    assert table in dataset.tasks, (
        f"no rdblearn task {table!r} in {db!r}; available: {list(dataset.tasks)}"
    )
    rdb_task = dataset.tasks[table]
    target_col = rdb_task.metadata.target_col

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
        f"{db}/{table}: {len(combined)} task rows vs {total_nodes} nodes in "
        f"table_info.json -- the data the features are for is not the data "
        f"that was preprocessed"
    )

    X = combined.drop(columns=[target_col])
    y = combined[target_col]

    base_model = LogisticRegression() if task_type == "clf" else Ridge()
    estimator = RDBLearnEstimator(
        base_estimator=make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0), base_model
        ),
        config=config,
    )
    estimator.fit(
        X=X,
        y=y,
        rdb=dataset.rdb,
        key_mappings=rdb_task.metadata.key_mappings,
        cutoff_time_column=rdb_task.metadata.time_col,
    )

    X_copy = X.copy()
    estimator._ensure_keys_are_strings(X_copy, estimator.key_mappings_)
    X_dfs = fastdfs.compute_dfs_features(
        estimator.rdb_,
        X_copy,
        key_mappings=estimator.key_mappings_,
        cutoff_time_column=estimator.cutoff_time_column_,
        config=estimator.config.dfs or DFSConfig(),
    )
    # RDBLearn's TabularPreprocessor hands its (tree-based) estimator the raw
    # entity key as an integer, the cutoff time as int64 nanoseconds (~1e18)
    # and the temporal diffs in nanoseconds (~1e17). Fed to TabICL as they are,
    # the id is noise, the absolute time extrapolates past every context row,
    # and the ~1e18 columns blow up the float32 per-context standardization
    # (2026-08-19 blobs: nMAE rose from 40 to 80 with context size while the
    # LightGBM arm on the same blobs was fine). So: drop the key and cutoff
    # columns and z-score every feature over all rows in float64, as the SQL
    # featurizer does. The preprocessor also expands the cutoff into
    # <cutoff>.year/.month/.day/.dayofweek; the year is absolute time again
    # (every context row precedes the query in it), and with it kept TabICL's
    # regression error still doubled from 256 to 8192 cells on the heavy-tailed
    # targets while LightGBM stayed flat (2026-08-27 probe: rel-amazon/item-ltv
    # raw MAE 9.6 -> 24.8 with the four columns, 8.9 -> 7.1 without; the SQL
    # features carry no calendar columns). Those expansions go with the cutoff.
    frame = estimator.preprocessor_.transform(X_dfs)
    keys = set(estimator.key_mappings_) | set(X.columns)
    keys.discard(estimator.cutoff_time_column_)
    cutoff = estimator.cutoff_time_column_
    dropped = [
        c
        for c in frame.columns
        if c in keys or c == cutoff or c.startswith(f"{cutoff}.")
    ]
    frame = frame.drop(columns=dropped)
    arr = frame.to_numpy(dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    mean = np.nanmean(arr, axis=0, keepdims=True)
    std = np.nanstd(arr, axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    arr = np.nan_to_num((arr - mean) / std, nan=0.0)
    feats = arr.astype(np.float32)
    assert feats.shape[0] == total_nodes
    assert np.isfinite(feats).all()
    print(f"[{db}] {table}: dropped {dropped}; kept {list(frame.columns)}", flush=True)

    feats.tofile(vectors_path)
    meta_path.write_text(
        json.dumps(
            {
                "n_features": feats.shape[1],
                "min_offset": min_offset,
                "total_nodes": total_nodes,
            }
        )
    )
    print(
        f"[{db}] {table}: {total_nodes} rows x {feats.shape[1]} features "
        f"in {time.time() - tic:.0f}s",
        flush=True,
    )
