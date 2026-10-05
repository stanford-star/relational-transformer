"""SQL-based featurizer for relbench tasks.

Computes hand-crafted SQL features via DuckDB against relbench data.
Features are pre-computed for all rows at init (global z-score normalization),
then looked up by node_idx in ``compute_features``.

In ``featurize``, filters train rows to those sharing the target's
foreign-key entity (like EntityFeaturizer), giving the downstream predictor
a more relevant training set.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import torch

from rt.rel2tab.featurizer import Featurizer, load_table_info
from rt.rel2tab.featurizers.sql_queries import SQL_REGISTRY


@dataclass
class SQLFeaturizerConfig:
    """Config for SQLFeaturizer.

    Args:
        pre_dir: Directory containing preprocessed table_info.json files.
        db_task_list: Path to the JSON db-task list naming the tasks to load
            data for.
        splits: Splits of those tasks to load.
    """

    pre_dir: str
    db_task_list: str
    splits: tuple[str, ...]

    def build(self, device):
        return SQLFeaturizer(
            pre_dir=self.pre_dir,
            db_task_list=self.db_task_list,
            splits=self.splits,
            db=None,
        )


def _run_sql_features(con, sql, task_df, key_cols):
    """Run feature SQL on task_df, return aligned feature array."""
    con.register("task_table", task_df)
    feats_df = con.execute(sql).df()
    feats_df = feats_df.drop_duplicates(subset=key_cols, keep="first")
    merged = task_df.merge(feats_df, on=key_cols, how="left")

    feat_cols = [c for c in feats_df.columns if c not in key_cols]
    if not feat_cols:
        return None, []

    arr = merged[feat_cols].values.astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    np.nan_to_num(arr, copy=False, nan=0.0)
    return arr, feat_cols


class SQLFeaturizer(Featurizer):
    """Pre-computed SQL features with entity-aware row filtering.

    At init:
      1. Loads relbench datasets into DuckDB.
      2. Runs feature SQL once for all rows per task.
      3. Z-score normalizes globally and stores as a contiguous tensor
         indexed by ``node_idx - min_offset``.

    In ``compute_features``: simple index lookup (no SQL).

    In ``featurize``: filters train rows to those sharing the target's
    foreign-key entity, falling back to all train rows if no matches.
    """

    def __init__(self, pre_dir, db_task_list, splits, db):
        from relbench.datasets import get_dataset
        from relbench.tasks import get_task

        from rt.data import get_tasks

        pre_dir = str(Path(pre_dir).expanduser())

        # Per-dataset DuckDB connections.
        connections: dict[str, duckdb.DuckDBPyConnection] = {}

        # (rt_db_name, rt_table_name) -> (features_tensor, min_offset)
        self._features: dict[tuple[str, str], tuple[torch.Tensor, int]] = {}

        all_tasks = get_tasks(pre_dir, db_task_list, splits)
        if db is not None:
            all_tasks = [t for t in all_tasks if db in t.db_name]

        seen: set[tuple[str, str]] = set()
        for task in all_tasks:
            key = (task.db_name, task.table_name)
            if key in seen:
                continue
            seen.add(key)

            ds = task.db_name.split("/")[-1]
            sql_key = (ds, task.table_name)
            if sql_key not in SQL_REGISTRY:
                raise ValueError(
                    f"No SQL features for {sql_key}."
                    f" Available: {list(SQL_REGISTRY.keys())}"
                )

            entry = SQL_REGISTRY[sql_key]
            sql = entry["sql"]
            entity_col = entry["entity_col"]
            time_col = entry["time_col"]

            # Load DB tables into DuckDB (once per dataset).
            if ds not in connections:
                con = duckdb.connect()
                con.execute("SET preserve_insertion_order=false")
                con.execute("SET threads=4")
                rb_dataset = get_dataset(ds, download=True)
                # this allows up-to-date rows in the context window, which matters for rel-f1
                rb_db = rb_dataset.get_db(upto_test_timestamp=False)
                for tbl_name, tbl in rb_db.table_dict.items():
                    con.register(tbl_name, tbl.df)
                connections[ds] = con
                print(
                    f"  SQLFeaturizer: loaded DB '{ds}' into DuckDB "
                    f"({len(rb_db.table_dict)} tables)"
                )

            con = connections[ds]

            # Build combined DataFrame aligned to node_idx offsets.
            table_info = load_table_info(task.db_name, pre_dir)
            splits_info: dict[str, dict] = {}
            for split in ["train", "val", "test"]:
                info_key = f"{task.table_name}:{split.capitalize()}"
                if info_key in table_info:
                    splits_info[split] = table_info[info_key]

            if not splits_info:
                raise KeyError(
                    f"Table '{task.table_name}' not in table_info.json for db "
                    f"'{task.db_name}'. Keys: {list(table_info.keys())}"
                )

            min_offset = min(info["node_idx_offset"] for info in splits_info.values())

            rb_task = get_task(ds, task.table_name, download=True)
            split_dfs = {
                "train": rb_task.get_table("train").df,
                "val": rb_task.get_table("val").df,
                "test": rb_task.get_table("test").df,
            }
            ordered = sorted(splits_info.items(), key=lambda x: x[1]["node_idx_offset"])
            parts = [split_dfs[s].reset_index(drop=True) for s, _ in ordered]
            combined_df = pd.concat(parts, ignore_index=True)

            # Pre-compute features for ALL rows.
            tic = time.time()
            key_cols = [time_col, entity_col]
            feat_array, feat_cols = _run_sql_features(con, sql, combined_df, key_cols)

            if feat_array is None:
                raise RuntimeError(
                    f"SQLFeaturizer: {ds}/{task.table_name} produced no features"
                )

            # Global z-score normalization.
            mean = feat_array.mean(axis=0, keepdims=True)
            std = feat_array.std(axis=0, keepdims=True)
            std = np.where(std < 1e-8, 1.0, std)
            feat_array = (feat_array - mean) / std

            feat_tensor = torch.from_numpy(feat_array)
            self._features[key] = (feat_tensor, min_offset)

            elapsed = time.time() - tic
            print(
                f"  SQLFeaturizer: precomputed {ds}/{task.table_name} "
                f"({len(combined_df)} rows, {len(feat_cols)} features, "
                f"{elapsed:.1f}s)"
            )

        # Free DuckDB connections (no longer needed after precomputation).
        for con in connections.values():
            con.close()

    def compute_features(self, task, node_idxs, device, batch_size):
        key = (task.db_name, task.table_name)
        feat_tensor, min_offset = self._features[key]
        local_idxs = node_idxs.cpu() - min_offset
        return feat_tensor[local_idxs].to(device)

    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        """Append per-row context features to the SQL features.

        For each row (train or test), computes:
          - same_entity: 1.0 if this row shares the target's FK entity, else 0.0
          - entity_mean: mean label of same-entity train rows
          - entity_count: log-scaled count of same-entity train rows
          - global_mean: mean label of all visible train rows
        """
        n_train = len(train_labels)
        if test_feat is None and (train_feats is None or n_train == 0):
            return train_feats, train_labels, test_feat

        # Everything runs on CPU (model.py moves tensors to CPU before
        # calling featurize / predict).
        global_mean = train_labels.mean().item() if n_train > 0 else 0.0

        # Entity match mask for train rows.
        target_match = (train_f2ps == target_f2p).all(dim=-1)
        n_match = target_match.sum().item()
        entity_mean = (
            train_labels[target_match].mean().item() if n_match > 0 else global_mean
        )
        log_entity_count = float(np.log1p(n_match))

        if train_feats is not None and n_train > 0:
            # Per-row: same_entity flag differs per train row.
            same_entity = target_match.float().unsqueeze(-1)
            shared = (
                torch.tensor(
                    [entity_mean, log_entity_count, global_mean],
                )
                .unsqueeze(0)
                .expand(n_train, -1)
            )
            train_feats = torch.cat([train_feats.cpu(), same_entity, shared], dim=-1)

        if test_feat is not None:
            test_ctx = torch.tensor(
                [1.0, entity_mean, log_entity_count, global_mean],
            )
            test_feat = torch.cat([test_feat.cpu(), test_ctx], dim=-1)

        return train_feats, train_labels, test_feat
