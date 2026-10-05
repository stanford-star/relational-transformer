"""
Pre-compute features for all rows in a database and save to disk.

Works with any :class:`~rt.rel2tab.featurizer.Featurizer` that implements
``compute_features()``.  The script builds the featurizer from the given
config, iterates over task tables, calls ``compute_features`` for every
node, and writes the resulting vectors to disk.

All unique databases the db-task list resolves to are processed in parallel
(one process per db).

Usage::

    python -m rt.rel2tab.featurize \\
        --featurize-batch-size 4096 --out-subdir rdblearn_features \\
        --num-workers 6 \\
        featurizer:rdb-learn-featurizer-config \\
        --featurizer.db-task-list data/db-task-lists/relbench-forecast.json \\
        --featurizer.splits test \\
        --featurizer.pre-dir ~/scratch/pre \\
        --featurizer.max-depth 2 \\
        --featurizer.max-train-samples 1000
"""

import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from rt.rel2tab.featurizers.rdblearn_featurizer import (
    RDBLearnFeaturizer,
    RDBLearnFeaturizerConfig,
)
from rt.rel2tab.featurizers.rt_featurizer import RTFeaturizer, RTFeaturizerConfig
from rt.rel2tab.featurizers.sql_featurizer import SQLFeaturizer, SQLFeaturizerConfig


@dataclass
class FeaturizeConfig:
    featurizer: RTFeaturizerConfig | RDBLearnFeaturizerConfig | SQLFeaturizerConfig
    featurize_batch_size: int
    out_subdir: str
    num_workers: int


def _build_featurizer(cfg, db, device):
    """Build a featurizer filtered to a single db."""
    if isinstance(cfg, RDBLearnFeaturizerConfig):
        return RDBLearnFeaturizer(
            pre_dir=cfg.pre_dir,
            db_task_list=cfg.db_task_list,
            splits=cfg.splits,
            max_depth=cfg.max_depth,
            max_train_samples=cfg.max_train_samples,
            db=db,
        )
    elif isinstance(cfg, SQLFeaturizerConfig):
        return SQLFeaturizer(
            pre_dir=cfg.pre_dir,
            db_task_list=cfg.db_task_list,
            splits=cfg.splits,
            db=db,
        )
    elif isinstance(cfg, RTFeaturizerConfig):
        return RTFeaturizer(
            embedder=cfg.embedder,
            d_text=cfg.d_text,
            num_blocks=cfg.num_blocks,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            d_ff=cfg.d_ff,
            compile=cfg.compile,
            materialize_attn_masks=cfg.materialize_attn_masks,
            load_ckpt_path=cfg.load_ckpt_path,
            device=device,
            db_task_list=cfg.db_task_list,
            splits=cfg.splits,
            pre_dir=cfg.pre_dir,
            ctx_size=cfg.ctx_size,
            bfs_width=cfg.bfs_width,
            shuffle_seed=cfg.shuffle_seed,
            context_seed=cfg.context_seed,
            vector_db_path=cfg.vector_db_path,
            db=db,
        )


def _featurize_db(featurizer_cfg, db, out_subdir, featurize_batch_size, local_rank):
    """Process all tables for a single db. Runs in a worker process."""
    from rt.data import get_tasks
    from rt.rel2tab.featurizer import (
        get_table_splits,
        load_table_info,
        validate_contiguous,
    )

    device = (
        f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    )

    if local_rank == 0:
        print(f"[{db}] Building featurizer on {device}...")
    featurizer = _build_featurizer(featurizer_cfg, db, device)

    pre_dir = featurizer_cfg.pre_dir
    tasks = get_tasks(pre_dir, featurizer_cfg.db_task_list, featurizer_cfg.splits)
    tasks = [t for t in tasks if db in t.db_name]
    if not tasks:
        if local_rank == 0:
            print(f"[{db}] No tasks found, skipping.")
        return

    table_info = load_table_info(db, pre_dir)

    # Deduplicate by table_name, keep first task per table.
    seen: set[str] = set()
    unique_tasks = []
    for t in tasks:
        if t.table_name not in seen:
            seen.add(t.table_name)
            unique_tasks.append(t)

    out_dir = Path(pre_dir).expanduser() / db / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in unique_tasks:
        table_name = task.table_name
        splits_info = get_table_splits(table_info, table_name)
        if not splits_info:
            if local_rank == 0:
                print(f"[{db}]   {table_name}: not in table_info (skipped)")
            continue
        validate_contiguous(splits_info, db, table_name)

        min_offset = min(info["node_idx_offset"] for info in splits_info.values())
        total_nodes = sum(info["num_nodes"] for info in splits_info.values())

        node_idxs = torch.arange(min_offset, min_offset + total_nodes)

        if local_rank == 0:
            print(f"[{db}]   {table_name}: {total_nodes} nodes ...")
        features = featurizer.compute_features(
            task, node_idxs, device, featurize_batch_size
        )

        if features is None:
            if local_rank == 0:
                print(f"[{db}]     -> no features produced (skipped)")
            continue

        vectors = features.cpu().float().numpy()
        vectors_path = out_dir / f"{table_name}_vectors.bin"
        meta_path = out_dir / f"{table_name}_meta.json"
        vectors.astype(np.float32).tofile(str(vectors_path))
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "n_features": vectors.shape[1],
                    "min_offset": min_offset,
                    "total_nodes": total_nodes,
                },
                f,
            )

        if local_rank == 0:
            vec_mb = vectors_path.stat().st_size / 1e6
            print(
                f"[{db}]     -> {vectors.shape[1]} features, "
                f"{vectors_path} ({vec_mb:.1f} MB)"
            )

    if local_rank == 0:
        print(f"[{db}] Done.")


def _worker(args):
    """Wrapper for multiprocessing — unpacks args tuple."""
    _featurize_db(*args)


def main(cfg: FeaturizeConfig):
    from rt.data import get_tasks

    if dist.is_initialized():
        global_rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        global_rank = 0
        local_rank = 0
    print(
        f"[rank init] global_rank={global_rank} local_rank={local_rank}",
        flush=True,
    )

    tasks = get_tasks(
        cfg.featurizer.pre_dir, cfg.featurizer.db_task_list, cfg.featurizer.splits
    )
    unique_dbs = sorted({t.db_name for t in tasks})

    if local_rank == 0:
        print(f"Found {len(unique_dbs)} databases: {unique_dbs}")
        print(f"Processing with {cfg.num_workers} workers...\n")

    worker_args = [
        (cfg.featurizer, db, cfg.out_subdir, cfg.featurize_batch_size, local_rank)
        for db in unique_dbs
    ]

    if cfg.num_workers <= 1:
        for args in worker_args:
            _featurize_db(*args)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=cfg.num_workers) as pool:
            pool.map(_worker, worker_args)

    if local_rank == 0:
        print("\nAll databases done.")


if __name__ == "__main__":
    import tyro

    main(tyro.cli(FeaturizeConfig))
