import json
import os
import uuid
from collections import OrderedDict
from pathlib import Path


def _atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp.{os.getpid()}.{uuid.uuid4().hex}.json"
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def main(
    *,
    method: str,
    device: str,
    db: str,
    table: str,
    split: str,
    pre_dir: str,
    features_root: str | None,
    out_dir: str,
    ctx_size_list: list[int],
    items_per_task: int,
    local_ctx_size: int,
    bfs_width: int,
    prefer_latest: bool,
    num_walks: int,
    walk_length: int,
    shuffle_seed: int,
    context_seed: int,
    tokens_per_gpu: int,
    num_workers: int,
    prefetch_factor: int,
    mmap_populate: bool,
    db_cutoff: str | int | None,
    vector_db_path: str | None,
    ckpt: str | None,
    tabicl_dir: str | None,
    tabicl_max_batch_size: int,
    tabicl_min_bin_size: int,
    tabicl_softmax_temperature: float,
    lgbm_n_jobs: int,
) -> None:
    out_path = Path(out_dir).expanduser() / f"{db}__{table}.json"
    if out_path.exists():
        print(f"{out_path} exists; nothing to do", flush=True)
        return

    import numpy as np
    import torch

    from rt.data import get_tasks
    from rt.eval import build_evaluator
    from rt.eval.metrics import metric_for

    (task,) = get_tasks(pre_dir, [(db, table)], (split,))
    ctx_sizes = sorted(ctx_size_list)

    if method == "rt":
        from rt.model import load_rt_model

        torch._dynamo.config.cache_size_limit = max(16, 2 * len(ctx_sizes))
        model, config = load_rt_model(ckpt, device=device, compile=device == "cuda")
        model = model.to(torch.bfloat16)
        embedder, d_text = config["embedder"], config["d_text"]
    else:
        from reproduce.baselines.rel2tab.model import Rel2TabModel
        from reproduce.baselines.rel2tab.precomputed import PrecomputedFeaturizer

        featurizer = PrecomputedFeaturizer(
            features_root,
            {
                "rdblearn_tabicl": "rdblearn_features",
                "rdblearn_lgbm": "rdblearn_features",
                "sql_tabicl": "sql_features",
                "sql_lgbm": "sql_features",
            }[method],
            [(db, table)],
        )
        if method.endswith("_tabicl"):
            from reproduce.baselines.rel2tab.tabicl_batched import (
                TabICLBatchedPredictor,
            )

            predictor = TabICLBatchedPredictor(
                max_batch_size=tabicl_max_batch_size,
                min_bin_size=tabicl_min_bin_size,
                softmax_temperature=tabicl_softmax_temperature,
                checkpoint_dir=tabicl_dir,
                device=device,
            )
        else:
            from reproduce.baselines.rel2tab.lgbm import LGBMPredictor

            predictor = LGBMPredictor(n_jobs=lgbm_n_jobs)
        model = Rel2TabModel(featurizer, predictor)
        embedder, d_text = "all-MiniLM-L12-v2", 384

    ev = build_evaluator(
        [task],
        pre_dir,
        embedder=embedder,
        d_text=d_text,
        device=device,
        ctx_size_list=ctx_sizes,
        local_ctx_size=local_ctx_size,
        bfs_width=bfs_width,
        prefer_latest=prefer_latest,
        num_walks=num_walks,
        walk_length=walk_length,
        tokens_per_gpu=tokens_per_gpu,
        items_per_task=items_per_task,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        context_seed=context_seed,
        shuffle_seed=shuffle_seed,
        mmap_populate=mmap_populate,
        vector_db_path=vector_db_path,
        db_cutoff=db_cutoff,
    )

    per_ctx: dict[int, dict] = OrderedDict()
    for _task, ctx, labels, preds_by_prefix, num_labels in ev.evaluate_raw(
        [(model, "")], ctx_sizes
    ):
        metric_name, metric_value = metric_for(
            task.task_type, labels, preds_by_prefix[""]
        )
        assert np.isfinite(metric_value), (
            f"{db}/{table} ctx={ctx}: {metric_name}={metric_value}"
        )
        per_ctx[int(ctx)] = {
            "metric_name": metric_name,
            "metric_value": metric_value,
            "n": int(labels.shape[0]),
            "mean_labels": float(np.mean(num_labels)),
        }
        print(
            f"{db}/{table} ctx={ctx}: {metric_name}={metric_value:.4f} "
            f"(n={labels.shape[0]}, labels={per_ctx[int(ctx)]['mean_labels']:.1f})",
            flush=True,
        )

    _atomic_write_json(
        out_path,
        {
            "method": method,
            "task": f"{db}/{table}",
            "db": db,
            "table": table,
            "task_type": task.task_type,
            "per_ctx": {str(c): per_ctx[c] for c in sorted(per_ctx)},
            "config": {
                "method": method,
                "device": device,
                "split": split,
                "ctx_sizes": ctx_sizes,
                "items_per_task": items_per_task,
                "local_ctx_size": local_ctx_size,
                "bfs_width": bfs_width,
                "prefer_latest": prefer_latest,
                "num_walks": num_walks,
                "walk_length": walk_length,
                "shuffle_seed": shuffle_seed,
                "context_seed": context_seed,
                "tokens_per_gpu": tokens_per_gpu,
                "db_cutoff": db_cutoff,
                "vector_db_path": vector_db_path,
                "pre_dir": pre_dir,
                "features_root": features_root,
                "ckpt": ckpt,
            },
        },
    )
    print(f"wrote {out_path}", flush=True)
