import contextlib
import json
import os
import random
import shutil
import signal
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb
from torch import optim
from torch.utils.data import DataLoader

from rt._env import _setup_env
from rt.data import TrainDataset, get_tasks, stage_paths
from rt.eval import Evaluator, member_context_seed, metric_for
from rt.model import (
    RelationalTransformer,
    load_model,
    resolve_checkpoint,
    save_model,
)
from rt.progress import fmt_bytes, fmt_duration, log
from rt.train.muon import Muon
from rt.train.swa import SwaState

BEST_METRICS = [("clf", "auroc", max), ("reg", "nmae", min)]


def setup_dist(num_workers: int = 0):
    _setup_env(num_workers)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            "nccl",
            timeout=timedelta(hours=1),
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        return f"cuda:{local_rank}", rank, local_rank, world_size, True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device, 0, 0, 1, False


def run_subdir(entity: str | None, project: str, run_id: str) -> Path:
    return Path(entity or "no-entity", project, run_id)


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def move(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.inference_mode()
def eval_avg_metrics(evaluators, nets_with_prefix, ctx_size_list):
    metric_names = {"clf": "auroc", "reg": "nmae"}
    acc = {}
    for evaluator in evaluators:
        for task, ctx, labels, preds_by_prefix, _nl in evaluator.evaluate_raw(
            nets_with_prefix, ctx_size_list
        ):
            for _, prefix in nets_with_prefix:
                key = (prefix, task.split, f"{task.db_name}/{task.table_name}", ctx)
                p = preds_by_prefix[prefix].astype(np.float64)
                if key not in acc:
                    acc[key] = [labels, np.zeros_like(p), task]
                acc[key][1] += p

    scores = {
        p: {
            s: {m: {} for m in metric_names.values()} for s in evaluators[0].eval_splits
        }
        for _, p in nets_with_prefix
    }
    for (prefix, split, task_key, _ctx), (labels, summed, task) in acc.items():
        try:
            _, v = metric_for(task.task_type, labels, summed / len(evaluators))
            v *= 100.0
        except ValueError:
            log(eval_metric_undefined=f"{prefix}/{split}/{task_key}")
            continue
        by_metric = scores[prefix].setdefault(
            split, {m: {} for m in metric_names.values()}
        )
        by_metric[metric_names[task.task_type]].setdefault(task_key, []).append(v)

    def _reduce(per_task):
        out = {k: float(np.mean(vs)) for k, vs in per_task.items()}
        out["mean"] = float(np.mean(list(out.values()))) if out else None
        return out

    return {
        p: {
            s: {m: _reduce(per_task) for m, per_task in by_metric.items()}
            for s, by_metric in by_split.items()
        }
        for p, by_split in scores.items()
    }


def main(
    *,
    embedder: str,
    d_text: int,
    num_blocks: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    compile: bool,
    materialize_attn_masks: bool,
    loss_fn: str,
    load_ckpt_path: str | None,
    db_task_list: list[tuple[str, str]] | str,
    train_splits: list[str],
    pre_dir: str,
    stage_dir: str | None,
    tokens_per_gpu: int,
    num_workers: int,
    prefetch_factor: int,
    ctx_size_list: list[int],
    local_ctx_size_list: list[int],
    bfs_width_list: list[int],
    prefer_latest_list: list[bool],
    num_walks: int,
    walk_length: int,
    mask_prob_max: float,
    items_per_task: int,
    delta_finetune: bool,
    optimizer: str,
    lr: float,
    wd: float,
    lr_warmup_steps: int,
    lr_decay_steps: int,
    grad_norm_max: float,
    total_bs: int,
    total_steps: int,
    early_stop_after_steps: int | None,
    can_select_init_model: bool,
    swa_momentum: float | None,
    seed: int,
    mmap_populate: bool,
    timeout_per_item: float,
    eval_freq: int | None,
    keep_all_ckpts: bool,
    vector_db_path: str | None,
    db_cutoff: str | int | None,
    eval_live: bool = True,
    resume_save_mins: float,
    eval_splits: list[str],
    eval_db_task_list: list[tuple[str, str]] | str,
    eval_pre_dir: str,
    eval_tokens_per_gpu: int,
    eval_num_workers: int,
    eval_prefetch_factor: int,
    eval_num_walks: int,
    eval_walk_length: int,
    eval_items_per_task: int,
    eval_ctx_size_list: list[int],
    eval_mmap_populate: bool,
    eval_shuffle_seed: int,
    eval_context_seed: int,
    eval_ensemble_size: int,
    eval_vector_db_path: str | None,
    eval_lcs_bw_pl_grid: list[tuple[int, int, bool]],
    eval_ctx_lcs_bw_pl_grid: list[tuple[int, int, int, bool]] | None = None,
    run_id: str,
    targets: dict[str, float],
    project: str,
    entity: str | None,
    run_name: str | None,
    wandb_disabled: bool,
    out_root: str,
) -> None:
    params = dict(locals())

    assert not (set(train_splits) & set(eval_splits)), (
        f"train_splits={train_splits} overlaps eval_splits={eval_splits}: an "
        f"evaluated split must not be trained on"
    )
    assert not eval_freq or eval_freq <= total_steps, (
        f"eval_freq={eval_freq} exceeds total_steps={total_steps}: the run would "
        "never evaluate, and would publish no checkpoint to select from"
    )
    assert early_stop_after_steps is None or "val" in eval_splits, (
        f"early_stop_after_steps={early_stop_after_steps} needs a val metric, "
        f"but eval_splits={eval_splits}"
    )

    assert 0 <= lr_decay_steps <= total_steps, (
        f"lr_decay_steps={lr_decay_steps} must fit inside total_steps={total_steps}"
    )

    start_tic = time.time()

    device, rank, local_rank, world_size, ddp = setup_dist(num_workers)
    is_main = rank == 0

    if stage_dir is not None:
        pre_dir, eval_pre_dir = stage_paths(
            stage_dir,
            [pre_dir, eval_pre_dir],
            local_rank=local_rank,
            barrier=torch.distributed.barrier if ddp else (lambda: None),
        )

    use_wandb = (not wandb_disabled) and is_main
    if use_wandb:
        job = os.environ.get("SLURM_JOB_ID")
        attempt = (
            f"{job}.{os.environ.get('SLURM_STEP_ID', '0')}"
            f".{os.environ.get('SLURM_RESTART_COUNT', '0')}"
            if job
            else f"{int(time.time())}"
        )
        wandb_dir = Path(out_root).expanduser() / run_subdir(entity, project, run_id)
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=project,
            entity=entity,
            name=f"{run_name}-{attempt}" if run_name else attempt,
            id=f"{run_id}-{attempt}",
            group=run_id,
            resume="never",
            config=params,
            dir=str(wandb_dir),
            settings=wandb.Settings(
                console_multipart=True,
                console_chunk_max_seconds=60,
            ),
        )
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    seed_everything(seed + rank)
    out_dir = Path(out_root).expanduser() / run_subdir(entity, project, run_id)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "params.json").write_text(
            json.dumps(params, indent=1, sort_keys=True) + "\n"
        )
        log(out_dir=out_dir)

    def build_net():
        return (
            RelationalTransformer(
                num_blocks=num_blocks,
                d_model=d_model,
                d_text=d_text,
                num_heads=num_heads,
                d_ff=d_ff,
                compile=compile,
                materialize_attn_masks=materialize_attn_masks,
                loss_fn=loss_fn,
            )
            .to(device)
            .to(torch.bfloat16)
        )

    net = build_net()
    raw_net = net
    if is_main:
        log(params=f"{sum(p.numel() for p in net.parameters()):_}")

    master = {
        n: p.detach().float().clone().requires_grad_(True)
        for n, p in raw_net.named_parameters()
    }

    @torch.no_grad()
    def net_to_master():
        for n, p in raw_net.named_parameters():
            master[n].copy_(p)

    @torch.no_grad()
    def master_to_net():
        for n, p in raw_net.named_parameters():
            p.copy_(master[n])

    def _is_muon(name, p):
        if p.ndim != 2 or min(p.shape) == 1:
            return False
        return not name.startswith(("enc_dict.", "dec_dict."))

    def _is_decayed(name, p):
        return p.ndim >= 2 and min(p.shape) > 1

    mem_params = torch.cuda.memory_allocated() if device.startswith("cuda") else 0

    named = list(master.items())
    muon_params = [p for n, p in named if _is_muon(n, p)]
    other_params = [p for n, p in named if not _is_muon(n, p)]
    assert len(muon_params) + len(other_params) == len(named)
    muon_decayed = [p for n, p in named if _is_muon(n, p) and _is_decayed(n, p)]
    assert len(muon_decayed) == len(muon_params), (
        "every parameter Muon takes is a weight matrix and so is decayed"
    )
    adamw_decayed = [p for n, p in named if not _is_muon(n, p) and _is_decayed(n, p)]
    adamw_plain = [p for n, p in named if not _is_muon(n, p) and not _is_decayed(n, p)]
    assert optimizer in ("muon", "adamw"), f"optimizer={optimizer!r}"
    assert not delta_finetune or load_ckpt_path is not None, (
        "delta_finetune has nothing to be a delta of without load_ckpt_path"
    )
    opt_wd = 0.0 if delta_finetune else wd
    adamw_kwargs = {
        "lr": lr,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "fused": device.startswith("cuda"),
    }
    if optimizer == "muon":
        opts = [
            Muon(
                muon_params,
                lr=lr,
                momentum=0.95,
                weight_decay=opt_wd,
                adjust_lr_fn="match_rms_adamw",
                ns_steps=5,
                compile=compile,
            ),
            optim.AdamW(
                [
                    {"params": adamw_decayed, "weight_decay": opt_wd},
                    {"params": adamw_plain, "weight_decay": 0.0},
                ],
                **adamw_kwargs,
            ),
        ]
    else:
        opts = [
            optim.AdamW(
                [
                    {"params": muon_decayed + adamw_decayed, "weight_decay": opt_wd},
                    {"params": adamw_plain, "weight_decay": 0.0},
                ],
                **adamw_kwargs,
            )
        ]

    def lr_lambda(step):
        warm = 1.0 if step >= lr_warmup_steps else (step + 1) / lr_warmup_steps
        left = total_steps - step
        decay = 1.0 if left >= lr_decay_steps else max(0.0, left / lr_decay_steps)
        return warm * decay

    scheds = [optim.lr_scheduler.LambdaLR(o, lr_lambda) for o in opts]

    kinds = (["live"] if eval_live else []) + (
        ["swa"] if swa_momentum is not None else []
    )
    assert kinds, "eval_live=False needs swa_momentum: nothing would be scored"
    best = {tt: {k: None for k in kinds} for tt, _, _ in BEST_METRICS}
    per_cfg = {tt: {} for tt, _, _ in BEST_METRICS}
    improved_at = 0
    start_step = 0
    resume_evaled_at = None

    resume_path = out_dir / "resume.pt"
    if load_ckpt_path is not None and not resume_path.exists():
        _, ckpt_path = resolve_checkpoint(load_ckpt_path)
        raw_net.load_state_dict(load_model(ckpt_path))
        net_to_master()
        if is_main:
            log(warm_started_from=load_ckpt_path)

    swa = (
        None
        if swa_momentum is None
        else SwaState(master.items(), momentum=swa_momentum)
    )
    swa_net = None if swa is None else build_net()

    delta_base = None
    if delta_finetune:
        _, base_path = resolve_checkpoint(load_ckpt_path)
        base_sd = load_model(base_path)
        delta_base = {
            n: base_sd[n].to(p.device, torch.float32)
            for n, p in master.items()
            if _is_decayed(n, p)
        }

    if resume_path.exists():
        ck = torch.load(resume_path, map_location="cpu", weights_only=True)
        assert "master" in ck, (
            f"{resume_path} carries no fp32 master weights; it predates them "
            "and cannot be resumed"
        )
        for n, t in master.items():
            t.data.copy_(ck["master"][n])
        master_to_net()
        for o, sd in zip(opts, ck["optimizers"], strict=True):
            o.load_state_dict(sd)
        for s, sd in zip(scheds, ck["schedulers"], strict=True):
            s.load_state_dict(sd)
        assert (ck["swa"] is None) == (swa is None), (
            f"{resume_path} was written with swa_momentum "
            f"{'set' if ck['swa'] is not None else 'None'}; this run has the other"
        )
        if swa is not None:
            swa.load_state_dict(ck["swa"])
        start_step = ck["step"]
        best = ck["best"]
        assert all(
            isinstance(v, dict) and set(v) == set(kinds) for v in best.values()
        ), (
            f"{resume_path} carries a per-task-type best tracker; it predates "
            "per-kind selection and cannot be resumed"
        )
        resume_evaled_at = ck.get("evaled_at", start_step)
        improved_at = ck.get("improved_at", start_step)
        if ck.get("per_cfg") is not None:
            per_cfg = ck["per_cfg"]
        if is_main:
            log(
                resumed_from=resume_path,
                step=start_step,
                world_size=world_size,
            )

    data_seed = seed
    train_init_tic = time.time()
    train_tasks = get_tasks(pre_dir, db_task_list, tuple(train_splits))
    train_ds = TrainDataset(
        tasks=train_tasks,
        pre_dir=pre_dir,
        train_ctx_size_list=ctx_size_list,
        train_tokens_per_gpu=tokens_per_gpu,
        total_bs=total_bs,
        global_rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_ctx_size_list=local_ctx_size_list,
        bfs_width_list=bfs_width_list,
        num_walks=num_walks,
        walk_length=walk_length,
        prefer_latest_list=prefer_latest_list,
        mask_prob_max=mask_prob_max,
        start_step=start_step,
        embedder=embedder,
        d_text=d_text,
        seed=data_seed,
        items_per_task=items_per_task,
        mask_prob_max_shared=None,
        mmap_populate=mmap_populate,
        timeout_per_item=timeout_per_item,
        vector_db_path=vector_db_path,
        db_cutoff=db_cutoff,
    )
    total_items = total_steps * total_bs
    stream_items = train_ds.num_items
    epochs_per_step = total_bs / stream_items
    if is_main:
        log(
            train_tasks_loaded=len(train_tasks),
            elapsed=fmt_duration(time.time() - train_init_tic),
            items=f"{total_items:_}",
            epochs=f"{total_items / stream_items:.2f}",
        )
    loader = DataLoader(
        train_ds,
        batch_size=None,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers else None,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    multi_ctx = len(ctx_size_list) > 1
    for c in ctx_size_list:
        tb = max(1, tokens_per_gpu // c)
        if total_bs < world_size * tb:
            assert total_bs % world_size == 0, (
                f"total_bs={total_bs} not divisible by world_size={world_size}"
                f" for ctx_size={c}"
            )
        else:
            assert total_bs % (world_size * tb) == 0, (
                f"total_bs={total_bs} must be divisible by world_size*train_bs="
                f"{world_size * tb} for ctx_size={c} (world_size={world_size}); "
                f"pick a GPU count dividing total_bs/train_bs={total_bs // tb}"
            )
    train_bs = max(1, tokens_per_gpu // max(ctx_size_list))
    if total_bs < world_size * train_bs:
        train_bs = max(1, total_bs // world_size)
        grad_accum = 1
    else:
        grad_accum = total_bs // (world_size * train_bs)

    accumulates = multi_ctx or grad_accum > 1
    if ddp:
        net = torch.nn.parallel.DistributedDataParallel(
            net,
            device_ids=[local_rank],
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
            forward_sync_buffers=False,
            static_graph=not accumulates,
        )

    assert len(set(map(tuple, eval_lcs_bw_pl_grid))) == len(eval_lcs_bw_pl_grid), (
        f"duplicate entries in eval_lcs_bw_pl_grid: {eval_lcs_bw_pl_grid}"
    )
    assert eval_ensemble_size >= 1, f"eval_ensemble_size={eval_ensemble_size}"
    val_tasks = get_tasks(eval_pre_dir, eval_db_task_list, tuple(eval_splits))

    def _member_seed(member):
        if eval_ensemble_size == 1:
            return eval_context_seed
        return member_context_seed(eval_context_seed, member)

    grid4 = eval_ctx_lcs_bw_pl_grid
    entries = (
        [(c, lcs, b, p) for (c, lcs, b, p) in grid4]
        if grid4
        else [(None, lcs, b, p) for (lcs, b, p) in eval_lcs_bw_pl_grid]
    )
    evaluators = (
        [
            (
                f"lcs{lcs}-bw{bw}-pl{int(pl)}_" if i else "",
                [ctx] if ctx is not None else eval_ctx_size_list,
                [
                    Evaluator(
                        tasks=val_tasks,
                        pre_dir=eval_pre_dir,
                        eval_bs=max(1, eval_tokens_per_gpu // max(eval_ctx_size_list)),
                        ctx_size_list=eval_ctx_size_list,
                        items_per_task=eval_items_per_task,
                        num_workers=eval_num_workers,
                        prefetch_factor=eval_prefetch_factor,
                        persistent_workers=eval_num_workers > 0,
                        local_ctx_size=lcs,
                        bfs_width=bw,
                        num_walks=eval_num_walks,
                        walk_length=eval_walk_length,
                        prefer_latest=pl,
                        mmap_populate=eval_mmap_populate,
                        legacy_boolean=False,
                        embedder=embedder,
                        d_text=d_text,
                        shuffle_seed=eval_shuffle_seed,
                        context_seed=_member_seed(member),
                        vector_db_path=eval_vector_db_path,
                        db_cutoff=db_cutoff,
                        global_rank=rank,
                        local_rank=local_rank,
                        world_size=world_size,
                        ddp=ddp,
                        device=device,
                    )
                    for member in range(eval_ensemble_size)
                ],
            )
            for i, (ctx, lcs, bw, pl) in enumerate(entries)
        ]
        if val_tasks
        else []
    )

    preempt = {"flag": False}

    def _on_signal(signum, frame):
        preempt["flag"] = True
        log(rank=rank, caught_signal=signum, action="save_next_step")

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGUSR1, _on_signal)

    if is_main:
        (out_dir / "config.json").write_text(
            json.dumps(
                {
                    "embedder": embedder,
                    "d_text": d_text,
                    "checkpoint_file": "model.safetensors",
                    "model": {
                        "num_blocks": num_blocks,
                        "d_model": d_model,
                        "d_text": d_text,
                        "num_heads": num_heads,
                        "d_ff": d_ff,
                        "materialize_attn_masks": materialize_attn_masks,
                        "loss_fn": loss_fn,
                    },
                },
                indent=2,
            )
            + "\n"
        )

    def save_resume(step):
        if not is_main:
            return
        tmp = out_dir / f"resume.pt.{os.getpid()}.tmp"
        try:
            with open(tmp, "wb") as f:
                torch.save(
                    {
                        "master": {n: t.detach().cpu() for n, t in master.items()},
                        "optimizers": [o.state_dict() for o in opts],
                        "schedulers": [s.state_dict() for s in scheds],
                        "swa": None if swa is None else swa.state_dict(),
                        "step": step,
                        "best": best,
                        "evaled_at": evaled_at,
                        "improved_at": improved_at,
                        "per_cfg": per_cfg,
                    },
                    f,
                )
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, resume_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def publish_latest(src, name):
        tmp = out_dir / f"{name}.{os.getpid()}.tmp"
        tmp.unlink(missing_ok=True)
        try:
            os.link(src, tmp)
            os.replace(tmp, out_dir / name)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def checkpoint(step):
        if not is_main:
            return
        live = out_dir / f"steps={step}.safetensors"
        save_model(raw_net.state_dict(), live, metadata={"step": step})
        publish_latest(live, "latest.safetensors")
        if swa is not None and swa.n > 0:
            swa.sync_to(swa_net.named_parameters())
            swa_ckpt = out_dir / f"swa_steps={step}.safetensors"
            save_model(
                swa_net.state_dict(),
                swa_ckpt,
                metadata={"step": step, "swa_n": swa.n},
            )
            publish_latest(swa_ckpt, "latest_swa.safetensors")

    def prune_ckpts(step):
        if keep_all_ckpts or not is_main:
            return
        keep = {
            f"{'swa_' if b['kind'] == 'swa' else ''}steps={b['step']}.safetensors"
            for by_kind in best.values()
            for b in by_kind.values()
            if b is not None
        } or {f"steps={step}.safetensors"} | (
            set() if swa is None else {f"swa_steps={step}.safetensors"}
        )
        for f in out_dir.glob("*steps=*.safetensors"):
            if f.name not in keep:
                f.unlink(missing_ok=True)

    def publish_best():
        if not is_main:
            return
        for tt, _, better in BEST_METRICS:
            by_kind = {k: b for k, b in best[tt].items() if b is not None}
            if not by_kind:
                log(skipped=f"best_{tt}", task_type=tt, reason="no_val_tasks")
                continue
            top = better(b["value"] for b in by_kind.values())
            overall = next(b for b in by_kind.values() if b["value"] == top)
            for label, b in [
                *((f"best_{kind}_{tt}", b) for kind, b in by_kind.items()),
                (f"best_{tt}", overall),
            ]:
                src = out_dir / (
                    f"swa_steps={b['step']}.safetensors"
                    if b["kind"] == "swa"
                    else f"steps={b['step']}.safetensors"
                )
                if not src.exists():
                    log(warning="best_ckpt_missing", label=label, expected=src)
                    continue
                dst = out_dir / f"{label}.safetensors"
                tmp = dst.with_suffix(f".{os.getpid()}.tmp")
                try:
                    shutil.copyfile(src, tmp)
                    os.replace(tmp, dst)
                except BaseException:
                    tmp.unlink(missing_ok=True)
                    raise
                log(
                    saved=label,
                    kind=b["kind"],
                    step=b["step"],
                    metric=b["metric"],
                    value=f"{b['value']:.4f}",
                    path=f"{label}.safetensors",
                )

    def consider(metrics, step):
        improved = False
        for kind in kinds:
            is_swa = kind == "swa"
            keys = [k for k in metrics if k.endswith("swa/") == is_swa]
            for tt, metric, better in BEST_METRICS:
                seen = []
                for k in keys:
                    if "val" not in metrics[k] or metric not in metrics[k]["val"]:
                        continue
                    x = metrics[k]["val"][metric].get("mean")
                    if x is None:
                        continue
                    seen.append(x)
                    cfg_key = (kind, k)
                    prev = per_cfg[tt].get(cfg_key)
                    if prev is None or better(x, prev) == x:
                        per_cfg[tt][cfg_key] = x
                        improved = True
                if not seen:
                    continue
                v = better(seen)
                cur = best[tt][kind]
                if cur is None or better(v, cur["value"]) == v:
                    best[tt][kind] = {
                        "kind": kind,
                        "step": step,
                        "value": v,
                        "metric": metric,
                    }
        return improved

    def run_eval(step):
        nonlocal improved_at
        if not evaluators:
            return False
        nets = [(raw_net, "")] if eval_live else []
        if swa is not None:
            swa.sync_to(swa_net.named_parameters())
            nets.append((swa_net, "swa/"))
        metrics = {}
        for tag, ctxs, members in evaluators:
            tagged_nets = [(n, tag + p) for n, p in nets]
            metrics.update(eval_avg_metrics(members, tagged_nets, ctxs))
        if is_main and (step > 0 or can_select_init_model) and consider(metrics, step):
            improved_at = step
        if is_main and use_wandb:
            wandb.log(
                {
                    "step": step,
                    "epoch": step * epochs_per_step,
                    **{
                        f"{p}{metric}/{split}/{task_key}": v
                        for p, by_split in metrics.items()
                        for split, by_metric in by_split.items()
                        for metric, per_task in by_metric.items()
                        for task_key, v in per_task.items()
                        if v is not None
                    },
                },
                step=step,
            )
        for n, _ in nets:
            n.train()
        if early_stop_after_steps is None:
            return False
        stop_early = is_main and step - improved_at >= early_stop_after_steps
        if ddp:
            flag = torch.tensor([1.0 if stop_early else 0.0], device=device)
            dist.broadcast(flag, src=0)
            stop_early = flag.item() > 0
        if stop_early and is_main:
            log(
                early_stop_at_step=step,
                last_improved_at=improved_at,
                patience=early_stop_after_steps,
            )
        return stop_early

    def reduce_step_stats(local_loss):
        if not ddp:
            return local_loss, preempt["flag"]
        stats = torch.tensor(
            [local_loss, 1.0 if preempt["flag"] else 0.0], device=device
        )
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        return stats[0].item() / world_size, stats[1].item() > 0

    it = iter(loader)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        mem_resident = torch.cuda.memory_allocated() - mem_params
        torch.cuda.reset_peak_memory_stats()

    step = start_step
    step_t0 = time.perf_counter()
    last_resume_t = time.perf_counter()
    evaled_at = resume_evaled_at if start_step > 0 else None
    is_cuda = device.startswith("cuda")

    if evaluators:
        evaluators[0][2][0].mem_guard([n for n in (raw_net, swa_net) if n is not None])
        if is_main:
            if is_cuda:
                torch.cuda.synchronize()
            log(
                mem_guard_passed=step,
                ctx=max(eval_ctx_size_list),
                peak=fmt_bytes(torch.cuda.max_memory_allocated()) if is_cuda else "-",
            )
    measure_mem = is_cuda
    while step < total_steps:
        if eval_freq and step % eval_freq == 0 and step != evaled_at:
            stop_early = run_eval(step)
            evaled_at = step
            checkpoint(step)
            if improved_at == step and any(
                b is not None for by_kind in best.values() for b in by_kind.values()
            ):
                publish_best()
            prune_ckpts(step)
            save_resume(step)
            if stop_early:
                break
            step_t0 = time.perf_counter()

        total_loss = 0.0
        load_time = 0.0
        if multi_ctx:
            t_load = time.perf_counter()
            micro_batches = next(it)
            load_time += time.perf_counter() - t_load
        else:
            micro_batches = None
        step_grad_accum = len(micro_batches) if multi_ctx else grad_accum
        for micro in range(step_grad_accum):
            if multi_ctx:
                raw_batch = micro_batches[micro]
            else:
                t_load = time.perf_counter()
                raw_batch = next(it)
                load_time += time.perf_counter() - t_load
            batch = move(raw_batch, device)
            sync = not (ddp and micro < step_grad_accum - 1)
            with contextlib.nullcontext() if sync else net.no_sync():
                out = net(batch, return_embeddings=False)
                loss = out[0] / step_grad_accum
                loss.backward()
            total_loss += loss.item()

        if measure_mem:
            torch.cuda.synchronize()
            mem_post_bwd = torch.cuda.memory_allocated()

        for n, p in raw_net.named_parameters():
            master[n].grad = None if p.grad is None else p.grad.float()
        norm = torch.nn.utils.get_total_norm(
            [t.grad for t in master.values() if t.grad is not None]
        )
        torch.nn.utils.clip_grads_with_norm_(master.values(), grad_norm_max, norm)
        for o in opts:
            o.step()
        if delta_base is not None and wd:
            factor = scheds[0].get_last_lr()[0] * wd
            with torch.no_grad():
                for n, t in master.items():
                    if n in delta_base:
                        t.add_(delta_base[n] - t, alpha=factor)
        master_to_net()
        if measure_mem:
            torch.cuda.synchronize()
            peak_step = torch.cuda.max_memory_allocated()
            mem_pre_zero = torch.cuda.memory_allocated()
        for o in opts:
            o.zero_grad(set_to_none=True)
        raw_net.zero_grad(set_to_none=True)
        if measure_mem:
            torch.cuda.synchronize()
            mem_grads = mem_pre_zero - torch.cuda.memory_allocated()
            mem_opt = mem_pre_zero - mem_post_bwd
            mem_fixed = mem_params + mem_resident
            mem_framework = mem_post_bwd - mem_fixed - mem_grads
            if is_main:
                log(
                    gpu_mem_peak=fmt_bytes(peak_step),
                    reserved=fmt_bytes(torch.cuda.max_memory_reserved()),
                    params=fmt_bytes(mem_params),
                    resident=fmt_bytes(mem_resident),
                    framework=fmt_bytes(mem_framework),
                    grads=fmt_bytes(mem_grads),
                    optimizer=fmt_bytes(mem_opt),
                    activations=fmt_bytes(peak_step - mem_fixed - mem_framework),
                )
            torch.cuda.reset_peak_memory_stats()
            measure_mem = False
        for s in scheds:
            s.step()
        if swa is not None:
            swa.update(master.items())
        step += 1

        total_loss, stop = reduce_step_stats(total_loss)

        step_time = time.perf_counter() - step_t0
        step_t0 = time.perf_counter()

        if is_main and step == start_step + 1:
            log(
                time_to_first_step=fmt_duration(time.time() - start_tic),
                compile_time=fmt_duration(step_time),
            )

        if is_main and use_wandb:
            wandb.log(
                {
                    "step": step,
                    "epoch": step * epochs_per_step,
                    "train/loss": total_loss,
                    "train/lr": scheds[0].get_last_lr()[0],
                    "train/grad_norm": float(norm),
                    "train/sec_per_step": step_time,
                    "train/load_time": load_time,
                    **{f"target/{k}": v for k, v in targets.items()},
                },
                step=step,
            )

        if time.perf_counter() - last_resume_t >= resume_save_mins * 60:
            save_resume(step)
            last_resume_t = time.perf_counter()
            step_t0 = time.perf_counter()
            if is_main:
                log(resume_saved_at_step=step, every_mins=resume_save_mins)

        if stop:
            if is_main:
                log(preempted_at_step=step, action="save_resume_and_exit")
            save_resume(step)
            if use_wandb:
                wandb.finish()
            if ddp:
                dist.barrier()
                dist.destroy_process_group()
            return

    if step != evaled_at:
        run_eval(step)
        evaled_at = step
        checkpoint(step)
        publish_best()
        prune_ckpts(step)
        save_resume(step)
    if is_main:
        log(load_with=f"rt.model.load_rt_model('{out_dir}/best_clf.safetensors')")
    if use_wandb:
        wandb.finish()
    if ddp:
        dist.destroy_process_group()
