import json
import math
import random
from functools import cache
from pathlib import Path

import ml_dtypes  # noqa: F401  # registers bfloat16 numpy dtype for rustler
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from rt.data.resolve import get_column_index, read_meta, resolve_pre_dir
from rt.rustler import Sampler

MAX_F2P_NBRS = 5


def _resolve_cutoff(db_cutoff, db_name: str, pre_dir: str) -> int | None:
    if db_cutoff is None:
        return None
    if isinstance(db_cutoff, int):
        return db_cutoff
    return _split_timestamps(db_name, pre_dir)[db_cutoff]


@cache
def _split_timestamps(db_name: str, pre_dir: str) -> dict[str, int]:
    try:
        import relbench
    except ImportError as e:
        raise ImportError(
            "a db_cutoff split needs the relbench package, which rt cannot "
            "declare as a dependency (git-only). Install it with:\n"
            '    pip install "relbench @ git+https://github.com/stanford-star/relbench@bac62c9"'
        ) from e

    source = read_meta(pre_dir, db_name).get("source")
    if not source:
        raise RuntimeError(
            f"{db_name}/meta.json has no 'source'; cannot read its split "
            f"timestamps from relbench, so db_cutoff cannot be honored"
        )
    dataset = relbench.load_dataset(source)
    return {
        "val": int(dataset.val_timestamp.timestamp()),
        "test": int(dataset.test_timestamp.timestamp()),
    }


def process_batch(tup, d_text):
    out = dict(tup)
    seq_len = out.pop("seq_len")

    for k, v in out.items():
        if k in [
            "number_values",
            "datetime_values",
            "text_values",
            "col_name_values",
        ]:
            out[k] = torch.from_numpy(v.view(np.float16)).view(torch.bfloat16)
        else:
            out[k] = torch.from_numpy(v)

    out["node_idxs"] = out["node_idxs"].view(-1, seq_len)
    out["sem_types"] = out["sem_types"].view(-1, seq_len)
    out["is_targets"] = out["is_targets"].view(-1, seq_len)
    out["is_task_nodes"] = out["is_task_nodes"].view(-1, seq_len)
    out["is_padding"] = out["is_padding"].view(-1, seq_len)
    out["table_name_idxs"] = out["table_name_idxs"].view(-1, seq_len)
    out["col_name_idxs"] = out["col_name_idxs"].view(-1, seq_len)
    out["class_value_idxs"] = out["class_value_idxs"].view(-1, seq_len)
    out["timestamps"] = out["timestamps"].view(-1, seq_len)
    out["seed_node_idxs"] = out["seed_node_idxs"].view(-1, seq_len)
    out["bfs_depths"] = out["bfs_depths"].view(-1, seq_len)

    out["f2p_nbr_idxs"] = out["f2p_nbr_idxs"].view(-1, seq_len, MAX_F2P_NBRS)
    out["number_values"] = out["number_values"].view(-1, seq_len, 1)
    out["datetime_values"] = out["datetime_values"].view(-1, seq_len, 1)
    out["text_values"] = out["text_values"].view(-1, seq_len, d_text)
    out["col_name_values"] = out["col_name_values"].view(-1, seq_len, d_text)

    return out


class RustlerDataset:
    def __init__(
        self,
        tasks,
        pre_dir: str,
        global_rank,
        local_rank,
        world_size,
        local_ctx_size_list: list[int],
        bfs_width_list: list[int],
        num_walks,
        walk_length,
        prefer_latest_list: list[bool],
        mask_prob_max,
        embedder,
        d_text,
        shuffle_seed,
        context_seed,
        items_per_task,
        quiet,
        ignore_data_errors,
        mmap_populate,
        legacy_boolean,
        timeout_per_item,
        vector_db_path: str | None,
        db_cutoff: str | int | None,
    ):
        pre_dir = resolve_pre_dir(pre_dir)
        if vector_db_path is not None:
            vector_db_path = str(Path(vector_db_path).expanduser())

        dataset_tuples = []
        target_column_indices = []
        drop_column_indices = []
        cutoff_timestamps = []
        skipped_tasks = []

        num_tasks = 0
        for task in tasks:
            num_tasks += 1
            db_name = task.db_name
            table_name = task.table_name
            target_column = task.target_column
            split = task.split
            columns_to_drop = task.leakage_columns
            try:
                if split == "train":
                    split = "Train"
                elif split == "val":
                    split = "Val"
                elif split == "test":
                    split = "Test"

                table_info_path = f"{pre_dir}/{db_name}/table_info.json"
                with open(table_info_path) as f:
                    table_info = json.load(f)

                table_info_key = (
                    f"{table_name}:Db"
                    if f"{table_name}:Db" in table_info
                    else f"{table_name}:{split}"
                )
                info = table_info[table_info_key]
                node_idx_offset = info["node_idx_offset"]
                num_nodes = info["num_nodes"]

                target_idx = get_column_index(
                    target_column, table_name, db_name, pre_dir
                )
                target_column_indices.append(target_idx)

                drop_indices = []
                for drop_table, col in columns_to_drop:
                    if drop_table == table_name and col == target_column:
                        continue
                    try:
                        drop_indices.append(
                            get_column_index(col, drop_table, db_name, pre_dir)
                        )
                    except ValueError:
                        pass
                drop_column_indices.append(drop_indices)

                cutoff_timestamps.append(_resolve_cutoff(db_cutoff, db_name, pre_dir))

                dataset_tuples.append((db_name, table_name, node_idx_offset, num_nodes))
            except Exception as e:
                if not ignore_data_errors:
                    raise
                task_name = f"{db_name}/{table_name}/{target_column}"
                skipped_tasks.append((task_name, e))

        if skipped_tasks and local_rank == 0 and not quiet:
            print(
                f"skipped {len(skipped_tasks)} of {num_tasks} tasks:",
                flush=True,
            )
            for task_name, e in skipped_tasks:
                print(f"  {task_name}: {e}", flush=True)

        self.world_size = world_size
        self.sampler = Sampler(
            dataset_tuples=dataset_tuples,
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
            local_ctx_size_list=local_ctx_size_list,
            bfs_width_list=bfs_width_list,
            num_walks=num_walks,
            walk_length=walk_length,
            prefer_latest_list=prefer_latest_list,
            mask_prob_max=mask_prob_max,
            embedder=embedder,
            pre_dir=pre_dir,
            d_text=d_text,
            shuffle_seed=shuffle_seed,
            context_seed=context_seed,
            target_columns=target_column_indices,
            columns_to_drop=drop_column_indices,
            cutoff_timestamps=cutoff_timestamps,
            items_per_task=items_per_task,
            quiet=quiet,
            ignore_data_errors=ignore_data_errors,
            num_prev_skipped=len(skipped_tasks),
            mmap_populate=mmap_populate,
            legacy_boolean=legacy_boolean,
            timeout_per_item=timeout_per_item,
            vector_db_path=vector_db_path,
        )
        self.num_items = self.sampler.num_items

        self.d_text = d_text

    def _process_batch(self, tup):
        return process_batch(tup, self.d_text)


class TrainDataset(RustlerDataset, IterableDataset):
    def __init__(
        self,
        tasks,
        pre_dir: str,
        train_ctx_size_list,
        train_tokens_per_gpu,
        total_bs,
        global_rank,
        local_rank,
        world_size,
        local_ctx_size_list: list[int],
        bfs_width_list: list[int],
        num_walks,
        walk_length,
        prefer_latest_list: list[bool],
        mask_prob_max,
        embedder,
        d_text,
        seed,
        items_per_task,
        mask_prob_max_shared,
        mmap_populate,
        timeout_per_item,
        vector_db_path: str | None,
        db_cutoff: str | int | None,
        start_step: int = 0,
    ):
        RustlerDataset.__init__(
            self,
            tasks=tasks,
            pre_dir=pre_dir,
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
            local_ctx_size_list=local_ctx_size_list,
            bfs_width_list=bfs_width_list,
            num_walks=num_walks,
            walk_length=walk_length,
            prefer_latest_list=prefer_latest_list,
            mask_prob_max=mask_prob_max,
            embedder=embedder,
            d_text=d_text,
            shuffle_seed=seed,
            context_seed=seed,
            items_per_task=items_per_task,
            quiet=False,
            ignore_data_errors=True,
            mmap_populate=mmap_populate,
            legacy_boolean=False,
            timeout_per_item=timeout_per_item,
            vector_db_path=vector_db_path,
            db_cutoff=db_cutoff,
        )
        self.train_ctx_size_list = train_ctx_size_list
        self.seed = random.Random(seed).getrandbits(64)
        self.start_step = start_step
        self.train_tokens_per_gpu = train_tokens_per_gpu
        self.total_bs = total_bs
        self.mask_prob_max_shared = mask_prob_max_shared
        for c in train_ctx_size_list:
            train_bs = max(1, train_tokens_per_gpu // c)
            if total_bs < world_size * train_bs:
                assert total_bs % world_size == 0, (
                    f"total_bs={total_bs} not divisible by world_size={world_size}"
                    f" for ctx_size={c}"
                )
            else:
                assert total_bs % (world_size * train_bs) == 0, (
                    f"total_bs={total_bs} not divisible by world_size*bs_per_gpu="
                    f"{world_size * train_bs} for ctx_size={c}"
                )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        stride = 1 if worker_info is None else worker_info.num_workers
        step, sampler_step = resume_positions(
            seed=self.seed,
            train_ctx_size_list=self.train_ctx_size_list,
            train_tokens_per_gpu=self.train_tokens_per_gpu,
            total_bs=self.total_bs,
            world_size=self.world_size,
            start_step=self.start_step,
            worker_id=worker_id,
            stride=stride,
        )
        self.sampler.set_step_py(sampler_step)
        self.sampler.set_stride_py(stride)
        single_ctx = len(self.train_ctx_size_list) == 1
        while True:
            if self.mask_prob_max_shared is not None:
                self.sampler.set_mask_prob_max_py(self.mask_prob_max_shared.value)
            train_ctx_size = random.Random(self.seed + step).choice(
                self.train_ctx_size_list
            )
            train_bs = max(1, self.train_tokens_per_gpu // train_ctx_size)
            if self.total_bs < self.world_size * train_bs:
                train_bs = max(1, self.total_bs // self.world_size)
                grad_accum = 1
            else:
                grad_accum = self.total_bs // (self.world_size * train_bs)
            if single_ctx:
                tup = self.sampler.batch_py(None, train_bs, train_ctx_size)
                yield self._process_batch(tup)
            else:
                batches = []
                for _ in range(grad_accum):
                    tup = self.sampler.batch_py(None, train_bs, train_ctx_size)
                    batches.append(self._process_batch(tup))
                yield batches
            step += stride


def resume_positions(
    *,
    seed: int,
    train_ctx_size_list: list[int],
    train_tokens_per_gpu: int,
    total_bs: int,
    world_size: int,
    start_step: int,
    worker_id: int,
    stride: int,
) -> tuple[int, int]:
    ctx_step = worker_id
    sampler_step = worker_id
    if start_step <= 0:
        return ctx_step, sampler_step
    single_ctx = len(train_ctx_size_list) == 1
    if single_ctx:
        ctx = train_ctx_size_list[0]
        train_bs = max(1, train_tokens_per_gpu // ctx)
        grad_accum = (
            1
            if total_bs < world_size * train_bs
            else total_bs // (world_size * train_bs)
        )
        target = start_step * grad_accum
        while ctx_step < target:
            ctx_step += stride
        return ctx_step, ctx_step
    while ctx_step < start_step:
        ctx = random.Random(seed + ctx_step).choice(train_ctx_size_list)
        train_bs = max(1, train_tokens_per_gpu // ctx)
        if total_bs < world_size * train_bs:
            grad_accum = 1
        else:
            grad_accum = total_bs // (world_size * train_bs)
        sampler_step += stride * grad_accum
        ctx_step += stride
    return ctx_step, sampler_step


class EvalDataset(Dataset):
    def __init__(
        self,
        rustler_dataset: RustlerDataset,
        eval_bs,
        eval_ctx_size,
    ):
        self.rustler_dataset = rustler_dataset
        self.eval_bs = eval_bs
        self.eval_ctx_size = eval_ctx_size

    def __len__(self):
        return math.ceil(
            self.rustler_dataset.num_items
            / (self.eval_bs * self.rustler_dataset.world_size)
        )

    def __getitem__(self, i):
        return self.rustler_dataset._process_batch(
            self.rustler_dataset.sampler.batch_py(i, self.eval_bs, self.eval_ctx_size)
        )


_TASK_TYPE = {"binary_classification": "clf", "regression": "reg"}

_SEM_TASK_TYPE = {"Boolean": "clf", "Number": "reg"}
