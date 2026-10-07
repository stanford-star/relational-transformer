import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

from rt.data import get_tasks
from rt.eval import main as evaluate
from rt.train import main as train
from rt.utils import log


CTX_SIZES = [128, 256, 512, 1024]


def lcs_bw_pl_grid() -> list[tuple[int, int, bool]]:
    return [
        (lcs, bw, pl)
        for lcs in (128, 256, 512, 1024)
        for bw in (16, 64, 256)
        for pl in (True, False)
    ]


def tune_configs() -> int:
    return sum(1 for lcs, _, _ in lcs_bw_pl_grid() for c in CTX_SIZES if lcs <= c)


def stage_dir(out_root: str, project: str, stage_id: str) -> Path:
    return Path(out_root).expanduser() / "no-wandb-entity" / project / stage_id


def rows(pre_dir: str, db: str, task: str, split: str) -> int:
    info = json.loads((Path(pre_dir).expanduser() / db / "table_info.json").read_text())
    return int(info[f"{task}:{split}"]["num_nodes"])


def finished(out_dir: Path, total_steps: int, patience_steps: int | None) -> bool:
    resume = out_dir / "resume.pt"
    if not resume.exists():
        return False
    ck = torch.load(resume, map_location="cpu", weights_only=True, mmap=True)
    if ck["step"] >= total_steps:
        return True
    return (
        patience_steps is not None
        and ck["evaled_at"] == ck["step"]
        and ck["step"] - ck["improved_at"] >= patience_steps
    )


def best_checkpoint(out_dir: Path, task_type: str) -> tuple[Path, int]:
    path = out_dir / f"best_swa_{task_type}.safetensors"
    assert path.exists(), f"{out_dir}: the selection arm published no {path.name}"
    with safe_open(path, framework="pt") as f:
        step = int(f.metadata()["step"])
    assert step > 0, f"{path}: selected step 0, which the outer arm cannot retrain"
    return path, step


def train_args(
    *,
    task_type: str,
    load_ckpt_path: str | None,
    db: str,
    task: str,
    train_splits: list[str],
    pre_dir: str,
    tokens_per_gpu: int,
    num_workers: int,
    eval_num_workers: int,
    total_steps: int,
    early_stop_after_steps: int | None,
    eval_freq: int | None,
    eval_splits: list[str],
    eval_rows: int,
    eval_ensemble_size: int,
    seed: int,
    project: str,
    run_id: str,
    run_name: str,
    out_root: str,
) -> dict:
    grid = lcs_bw_pl_grid()
    return {
        "embedder": "all-MiniLM-L12-v2",
        "d_text": 384,
        "num_blocks": 12,
        "d_model": 512,
        "num_heads": 8,
        "d_ff": 2048,
        "compile": True,
        "materialize_attn_masks": True,
        "loss_fn": {"clf": "bce", "reg": "l1"}[task_type],
        "load_ckpt_path": load_ckpt_path,
        "db_task_list": [(db, task)],
        "train_splits": train_splits,
        "pre_dir": pre_dir,
        "stage_dir": None,
        "tokens_per_gpu": tokens_per_gpu,
        "num_workers": num_workers,
        "prefetch_factor": 2,
        "ctx_size_list": [128, 256, 512, 1024],
        "local_ctx_size_list": sorted({lcs for lcs, _, _ in grid}),
        "bfs_width_list": sorted({bw for _, bw, _ in grid}),
        "prefer_latest_list": sorted({pl for _, _, pl in grid}),
        "num_walks": 1_000,
        "walk_length": 20,
        "mask_prob_max": 0.0,
        "items_per_task": 1_000_000_000,
        "delta_finetune": load_ckpt_path is not None,
        "optimizer": "muon",
        "lr": 5e-4,
        "wd": 0.1,
        "lr_warmup_steps": 0,
        "lr_decay_steps": 0,
        "grad_norm_max": 1.0,
        "total_bs": 256,
        "total_steps": total_steps,
        "early_stop_after_steps": early_stop_after_steps,
        "can_select_init_model": False,
        "swa_momentum": 0.9999,
        "seed": seed,
        "mmap_populate": True,
        "timeout_per_item": 10.0,
        "eval_freq": eval_freq,
        "keep_all_ckpts": False,
        "vector_db_path": None,
        "db_cutoff": None,
        "eval_live": False,
        "resume_save_mins": 20.0,
        "eval_splits": eval_splits,
        "eval_db_task_list": [(db, task)],
        "eval_pre_dir": pre_dir,
        "eval_tokens_per_gpu": 2**18,
        "eval_num_workers": eval_num_workers,
        "eval_prefetch_factor": 2,
        "eval_num_walks": 1_000,
        "eval_walk_length": 20,
        "eval_items_per_task": eval_rows,
        "eval_ctx_size_list": [1024],
        "eval_mmap_populate": True,
        "eval_shuffle_seed": 0,
        "eval_context_seed": 1,
        "eval_ensemble_size": eval_ensemble_size,
        "eval_vector_db_path": None,
        "eval_lcs_bw_pl_grid": [(1024, 256, False)],
        "eval_ctx_lcs_bw_pl_grid": [(1024, 1024, 256, False), (128, 128, 16, True)],
        "run_id": run_id,
        "targets": {},
        "project": project,
        "wandb_entity": None,
        "run_name": run_name,
        "wandb_disabled": True,
        "out_root": out_root,
    }


def eval_args(
    *,
    load_ckpt_path: str,
    db: str,
    task: str,
    pre_dir: str,
    num_workers: int,
    splits: list[str],
    val_items: int | None,
    test_items: int | None,
    ctx_size_list: list[int],
    grid: list[tuple[int, int, bool]],
    val_ensemble_size: int,
    test_ensemble_size: int,
    shuffle_seed: int,
    project: str,
    run_id: str,
    run_name: str,
    out_root: str,
) -> dict:
    return {
        "load_ckpt_path": load_ckpt_path,
        "embedder": "all-MiniLM-L12-v2",
        "d_text": 384,
        "num_blocks": 12,
        "d_model": 512,
        "num_heads": 8,
        "d_ff": 2048,
        "splits": splits,
        "db_task_list": [(db, task)],
        "pre_dir": pre_dir,
        "tokens_per_gpu": 2**18,
        "num_workers": num_workers,
        "prefetch_factor": 2,
        "num_walks": 1_000,
        "walk_length": 20,
        "val_items_per_task": val_items,
        "test_items_per_task": test_items,
        "ctx_size_list": ctx_size_list,
        "mmap_populate": True,
        "shuffle_seed": shuffle_seed,
        "context_seed": 0,
        "vector_db_path": None,
        "db_cutoff": None,
        "lcs_bw_pl_grid": grid,
        "val_ensemble_size": val_ensemble_size,
        "test_ensemble_size": test_ensemble_size,
        "run_id": run_id,
        "run_name": run_name,
        "targets": {},
        "project": project,
        "wandb_entity": None,
        "out_root": out_root,
        "wandb_disabled": True,
    }


def main(
    *,
    model: str,
    db: str,
    task: str,
    load_ckpt_root: str | None,
    pre_dir: str,
    tokens_per_gpu: int,
    num_workers: int,
    eval_num_workers: int,
    selection_steps: int,
    patience_steps: int,
    eval_freq: int,
    eval_rows: int,
    selection_ensemble_size: int,
    tune_rows: int,
    test_ensemble_size: int,
    seed: int,
    project: str,
    out_root: str,
) -> None:
    (rt_task,) = get_tasks(pre_dir, [(db, task)], ("train",))
    task_type = rt_task.task_type
    warm_start = load_ckpt_root
    name = f"{model}/{db}/{task}"

    def stage(s: str) -> tuple[str, Path]:
        stage_id = f"{model}-{db}-{task}-{s}"
        return stage_id, stage_dir(out_root, project, stage_id)

    inner_id, inner_dir = stage("inner")
    if not finished(inner_dir, selection_steps, patience_steps):
        log(stage=inner_id, warm_start=warm_start, total_steps=selection_steps)
        train(
            **train_args(
                task_type=task_type,
                load_ckpt_path=warm_start,
                db=db,
                task=task,
                train_splits=["train"],
                pre_dir=pre_dir,
                tokens_per_gpu=tokens_per_gpu,
                num_workers=num_workers,
                eval_num_workers=eval_num_workers,
                total_steps=selection_steps,
                early_stop_after_steps=patience_steps,
                eval_freq=eval_freq,
                eval_splits=["val"],
                eval_rows=eval_rows,
                eval_ensemble_size=selection_ensemble_size,
                seed=seed,
                project=project,
                run_id=inner_id,
                run_name=f"{name}/inner",
                out_root=out_root,
            )
        )
        assert finished(inner_dir, selection_steps, patience_steps), (
            f"{inner_id}: the selection arm returned before it finished; rerun "
            f"this job and it resumes from {inner_dir / 'resume.pt'}"
        )
    checkpoint, step = best_checkpoint(inner_dir, task_type)
    log(stage=inner_id, selected_step=step, checkpoint=checkpoint)

    tune_id, tune_dir = stage("tune")
    tuning = tune_dir / "tuning.json"
    if not tuning.exists():
        log(stage=tune_id, configs=tune_configs(), rows=tune_rows)
        evaluate(
            **eval_args(
                load_ckpt_path=str(checkpoint),
                db=db,
                task=task,
                pre_dir=pre_dir,
                num_workers=num_workers,
                splits=["val"],
                val_items=tune_rows,
                test_items=None,
                ctx_size_list=CTX_SIZES,
                grid=lcs_bw_pl_grid(),
                val_ensemble_size=selection_ensemble_size,
                test_ensemble_size=1,
                shuffle_seed=1,
                project=project,
                run_id=tune_id,
                run_name=f"{name}/tune",
                out_root=out_root,
            )
        )
        assert tuning.exists(), f"{tune_id}: the context search wrote no {tuning}"
    tuned = json.loads(tuning.read_text())[f"{db}/{task}"]
    ctx, lcs, bw, pl = tuned["best_cfg"]
    log(
        stage=tune_id,
        context=str((ctx, lcs, bw, pl)),
        value=f"{tuned['best_value']:.4f}",
    )

    train_rows = rows(pre_dir, db, task, "Train")
    val_rows = rows(pre_dir, db, task, "Val")
    refit_steps = math.ceil(step * (train_rows + val_rows) / train_rows)
    (inner_dir / "selection.json").write_text(
        json.dumps(
            {
                "step": step,
                "checkpoint": str(checkpoint),
                "train_rows": train_rows,
                "val_rows": val_rows,
                "refit_steps": refit_steps,
                "context": [ctx, lcs, bw, pl],
                "tune_value": tuned["best_value"],
            },
            indent=1,
        )
        + "\n"
    )

    outer_id, outer_dir = stage("outer")
    if not finished(outer_dir, refit_steps, None):
        log(stage=outer_id, total_steps=refit_steps, scaled_from=step)
        train(
            **train_args(
                task_type=task_type,
                load_ckpt_path=warm_start,
                db=db,
                task=task,
                train_splits=["train", "val"],
                pre_dir=pre_dir,
                tokens_per_gpu=tokens_per_gpu,
                num_workers=num_workers,
                eval_num_workers=eval_num_workers,
                total_steps=refit_steps,
                early_stop_after_steps=None,
                eval_freq=None,
                eval_splits=[],
                eval_rows=eval_rows,
                eval_ensemble_size=1,
                seed=seed,
                project=project,
                run_id=outer_id,
                run_name=f"{name}/outer",
                out_root=out_root,
            )
        )
        assert finished(outer_dir, refit_steps, None), (
            f"{outer_id}: the reporting arm returned before it finished; rerun "
            f"this job and it resumes from {outer_dir / 'resume.pt'}"
        )
    final = outer_dir / "latest_swa.safetensors"
    assert final.exists(), f"{outer_dir}: the reporting arm left no {final.name}"

    test_id, test_dir = stage("test")
    csv = test_dir / "eval_out" / f"{db}__{task}.csv"
    if not csv.exists():
        log(stage=test_id, seeds=test_ensemble_size, context=str((ctx, lcs, bw, pl)))
        test_dir.mkdir(parents=True, exist_ok=True)
        evaluate(
            **eval_args(
                load_ckpt_path=str(final),
                db=db,
                task=task,
                pre_dir=pre_dir,
                num_workers=num_workers,
                splits=["test"],
                val_items=None,
                test_items=1_000_000_000,
                ctx_size_list=[ctx],
                grid=[(lcs, bw, pl)],
                val_ensemble_size=1,
                test_ensemble_size=test_ensemble_size,
                shuffle_seed=0,
                project=project,
                run_id=test_id,
                run_name=f"{name}/test",
                out_root=out_root,
            )
        )
        assert csv.exists(), f"{test_id}: the test ensemble wrote no {csv}"
    log(done=name, prediction_table=csv)
