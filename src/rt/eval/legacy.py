from pathlib import Path

import torch

from rt.data import get_tasks
from rt.eval._eval import build_evaluator
from rt.eval.metrics import metric_for
from rt.eval.relbench import _emit_and_score
from rt.model.legacy._common import LEGACY_EMBEDDER


def run(
    model_for_task,
    *,
    out_dir: str,
    pre_dir: str,
    db_task_list: str,
    ctx_size: int,
    local_ctx_size: int,
    num_walks: int,
    walk_length: int,
    bfs_width: int,
    prefer_latest: bool,
    tokens_per_gpu: int,
    items_per_task: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle_seed: int,
    context_seed: int,
    mmap_populate: bool,
    vector_db_path: str | None,
    db_cutoff: str | None,
) -> dict:
    """``model_for_task(task) -> nn.Module`` supplies the (possibly per-task)
    legacy checkpoint, already on-device in bf16."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir).expanduser()

    tasks = get_tasks(pre_dir, db_task_list, ("test",))
    if not tasks:
        raise SystemExit(f"no tasks found in {pre_dir}")

    by_metric: dict[str, list[float]] = {}
    results = {}
    print(f"\n{'task':40} {'metric':8} {'value':>9} {'n':>7}  {'align':>11}  debug")
    for task in tasks:
        model = model_for_task(task).to(device).to(torch.bfloat16)
        ev = build_evaluator(
            [task],
            pre_dir,
            embedder=LEGACY_EMBEDDER,
            d_text=384,
            device=device,
            ctx_size=ctx_size,
            local_ctx_size=local_ctx_size,
            bfs_width=bfs_width,
            prefer_latest=prefer_latest,
            num_walks=num_walks,
            walk_length=walk_length,
            tokens_per_gpu=tokens_per_gpu,
            items_per_task=items_per_task,
            num_workers=num_workers,
            shuffle_seed=shuffle_seed,
            context_seed=context_seed,
            mmap_populate=mmap_populate,
            prefetch_factor=prefetch_factor,
            vector_db_path=vector_db_path,
            db_cutoff=db_cutoff,
            legacy_boolean=True,
        )
        for _task, _ctx, labels, preds_by_prefix, _nl, node_idxs in ev.evaluate_raw(
            [(model, "")], [ctx_size], with_node_idxs=True
        ):
            preds = preds_by_prefix[""]
            mname, mval, n, align, _ = _emit_and_score(
                out_dir,
                task,
                pre_dir,
                LEGACY_EMBEDDER,
                labels,
                preds,
                node_idxs,
            )
            nm, nv = metric_for(task.task_type, labels, preds)
            by_metric.setdefault(mname, []).append(mval)
            results[f"{task.db_name}/{task.table_name}"] = {
                "metric": mname,
                "value": mval,
                "n": n,
            }
            print(
                f"{task.db_name + '/' + task.table_name:40} {mname:8} {mval:>9.4f} "
                f"{n:>7}  {align:>11}  norm[{nm}]={nv:.4f}",
                flush=True,
            )
        del model, ev
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'mean':40}")
    for name, vals in by_metric.items():
        print(f"  {name:10} {sum(vals) / len(vals):>9.4f}  (over {len(vals)} tasks)")
    print(
        f"\nsubmission CSVs written to {out_dir}/  "
        f"(validate: python -m relbench.submit {out_dir})"
    )
    return results
