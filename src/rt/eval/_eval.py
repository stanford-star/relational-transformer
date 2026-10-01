import json
import os
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb

from rt._env import _setup_env
from rt.data import get_tasks
from rt.eval.evaluator import Evaluator
from rt.eval.metrics import metric_for
from rt.eval.relbench import _emit_and_score
from rt.model import load_rt_model
from rt.progress import log

METRIC_NAMES = {"clf": "auroc", "reg": "nmae"}


def setup_dist(num_workers: int = 0):
    _setup_env(num_workers)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            "nccl",
            timeout=timedelta(minutes=20),
            device_id=torch.device(f"cuda:{local_rank}"),
        )
        return f"cuda:{local_rank}", rank, local_rank, world_size, True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device, 0, 0, 1, False


def check_embedder(config: dict, embedder: str, d_text: int) -> None:
    assert "embedder" in config, (
        f"checkpoint config.json has no embedder; it cannot be evaluated, "
        f"because the text embeddings read from pre_dir are selected by name "
        f"({embedder} here) and a wrong choice is silently wrong, not an error"
    )
    assert config["embedder"] == embedder, (
        f"embedder mismatch: config={embedder} checkpoint={config['embedder']}. "
        f"The context is embedded with {embedder}'s vectors from pre_dir while "
        f"the weights were trained against {config['embedder']}'s, which scores "
        f"garbage rather than failing. Pass embedder={config['embedder']!r} and "
        f"the matching d_text, or evaluate a checkpoint trained on {embedder}"
    )
    assert config["d_text"] == d_text, (
        f"d_text mismatch: config={d_text} checkpoint={config['d_text']} for "
        f"embedder {embedder}"
    )


def main(
    *,
    load_ckpt_path: str,
    embedder: str,
    d_text: int,
    num_blocks: int,
    d_model: int,
    num_heads: int,
    d_ff: int,
    splits: list[str],
    db_task_list: list[tuple[str, str]] | str,
    pre_dir: str,
    tokens_per_gpu: int,
    num_workers: int,
    prefetch_factor: int,
    num_walks: int,
    walk_length: int,
    val_items_per_task: int | None,
    test_items_per_task: int | None,
    ctx_size_list: list[int],
    mmap_populate: bool,
    shuffle_seed: int,
    context_seed: int,
    vector_db_path: str | None,
    db_cutoff: str | int | None,
    lcs_bw_pl_grid: list[tuple[int, int, bool]],
    val_ensemble_size: int,
    test_ensemble_size: int,
    run_id: str,
    run_name: str | None,
    targets: dict[str, float],
    project: str,
    entity: str | None,
    out_root: str,
    wandb_disabled: bool,
) -> None:
    params = dict(locals())
    grid = [tuple(cfg) for cfg in lcs_bw_pl_grid]
    ctx_sizes = sorted(ctx_size_list)
    assert ctx_sizes, "nothing to evaluate at"
    n_cfgs = sum(1 for lcs, _, _ in grid for c in ctx_sizes if lcs <= c)
    assert n_cfgs, "every lcs exceeds every ctx size; no configuration survives"
    assert val_ensemble_size >= 1 and test_ensemble_size >= 1, "sizes are seed counts"
    assert n_cfgs > 1 or val_ensemble_size == 1, (
        "a one-configuration grid tunes nothing, so val_ensemble_size buys nothing"
    )
    assert (
        wandb_disabled or n_cfgs > 1 or (test_ensemble_size > 1 and "test" in splits)
    ), "the wandb curves here are val metric vs tune/idx and test metric vs ens_size"
    assert wandb_disabled or targets, "nothing to draw the curve against"
    csv_out_dir = (
        Path(out_root).expanduser()
        / (entity or "no-entity")
        / project
        / run_id
        / "eval_out"
    )
    device, global_rank, local_rank, world_size, ddp = setup_dist(num_workers)

    use_wandb = (not wandb_disabled) and global_rank == 0
    if use_wandb:
        job = os.environ.get("SLURM_JOB_ID")
        attempt = (
            f"{job}.{os.environ.get('SLURM_RESTART_COUNT', '0')}"
            if job
            else f"{int(time.time())}"
        )
        csv_out_dir.parent.mkdir(parents=True, exist_ok=True)
        wandb.init(
            project=project,
            entity=entity,
            name=f"{run_name}-{attempt}" if run_name else attempt,
            id=f"{run_id}-{attempt}",
            group=run_id,
            resume="never",
            config=params,
            dir=str(csv_out_dir.parent),
            settings=wandb.Settings(
                console_multipart=True,
                console_chunk_max_seconds=60,
            ),
        )
        wandb.define_metric("ens_size")
        wandb.define_metric("tune/idx")
        wandb.define_metric("*", step_metric="ens_size")
        wandb.define_metric("tune/*", step_metric="tune/idx")

    checkpoint = load_ckpt_path
    assert checkpoint is not None, "model.load_ckpt_path is required"
    net, config = load_rt_model(checkpoint, device=device, compile=False)
    net = net.to(torch.bfloat16)
    check_embedder(config, embedder, d_text)
    if global_rank == 0:
        log(
            model_loaded=config.get("name", checkpoint),
            embedder=embedder,
            device=device,
            world_size=world_size,
        )
    if global_rank == 0 and config.get("task_type") in ("clf", "reg"):
        log(
            warning="ckpt_task_type_mismatch",
            ckpt_task_type=config["task_type"],
            evaluated_on="clf_and_reg",
        )
    ckpt_model = config.get("model", {})
    mismatches = [
        f"{k}: config={v} checkpoint={ckpt_model[k]}"
        for k, v in (
            ("num_blocks", num_blocks),
            ("d_model", d_model),
            ("d_text", d_text),
            ("num_heads", num_heads),
            ("d_ff", d_ff),
        )
        if k in ckpt_model and ckpt_model[k] != v
    ]
    if mismatches and global_rank == 0:
        log(
            warning="model_config_ignored",
            mismatches=";".join(mismatches).replace(" ", "_"),
        )

    eval_kwargs = {
        "embedder": embedder,
        "d_text": d_text,
        "device": device,
        "num_walks": num_walks,
        "walk_length": walk_length,
        "tokens_per_gpu": tokens_per_gpu,
        "num_workers": num_workers,
        "shuffle_seed": shuffle_seed,
        "prefetch_factor": prefetch_factor,
        "mmap_populate": mmap_populate,
        "vector_db_path": vector_db_path,
        "db_cutoff": db_cutoff,
        "global_rank": global_rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "ddp": ddp,
    }
    assert "val" not in splits or val_items_per_task is not None, (
        "no val_items_per_task for the val split"
    )
    assert "test" not in splits or test_items_per_task is not None, (
        "no test_items_per_task for the test split"
    )
    if n_cfgs > 1 or test_ensemble_size > 1:
        assert n_cfgs == 1 or val_items_per_task is not None, (
            "tuning reads the val split whatever `splits` says, so it needs a "
            "val_items_per_task"
        )
        tune_only = "test" not in splits
        assert not (tune_only and n_cfgs == 1), (
            "one configuration has nothing to tune; ask for the test split"
        )
        val_tasks = get_tasks(pre_dir, db_task_list, ("val",)) if n_cfgs > 1 else None
        test_tasks = [] if tune_only else get_tasks(pre_dir, db_task_list, ("test",))
        if not test_tasks and not tune_only:
            raise SystemExit(f"no tasks found in {pre_dir}")
        run_ensemble(
            net,
            pre_dir,
            val_tasks,
            test_tasks,
            grid=grid,
            ctx_sizes=ctx_sizes,
            val_items=val_items_per_task,
            test_items=test_items_per_task,
            context_seed=context_seed,
            val_ensemble_size=val_ensemble_size,
            test_ensemble_size=test_ensemble_size,
            tune_only=tune_only,
            tuning_out_path=csv_out_dir.parent / "tuning.json",
            resume_path=csv_out_dir.parent / "ensemble_resume.pt",
            csv_out_dir=csv_out_dir,
            targets=targets,
            use_wandb=use_wandb,
            **eval_kwargs,
        )
        if use_wandb:
            wandb.finish()
        _teardown_dist(ddp)
        return

    tasks = get_tasks(pre_dir, db_task_list, tuple(splits))
    if not tasks:
        raise SystemExit(f"no tasks found in {pre_dir}")
    (ctx_size,) = ctx_sizes
    lcs, bw, pl = grid[0]
    (items,) = {
        {"val": val_items_per_task, "test": test_items_per_task}[s] for s in splits
    }
    ev = build_evaluator(
        tasks,
        pre_dir,
        items_per_task=items,
        ctx_size_list=[ctx_size],
        local_ctx_size=lcs,
        bfs_width=bw,
        prefer_latest=pl,
        context_seed=context_seed,
        **eval_kwargs,
    )
    run_and_report(
        net,
        tasks,
        pre_dir,
        ctx_size=ctx_size,
        csv_out_dir=csv_out_dir,
        evaluator=ev,
        embedder=embedder,
    )
    _teardown_dist(ddp)


def _teardown_dist(ddp):
    if ddp:
        dist.barrier()
        dist.destroy_process_group()


def member_context_seed(context_seed: int, member: int) -> int:
    mask = (1 << 64) - 1
    z = (context_seed + 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return ((z ^ (z >> 31)) + member) & mask


def build_evaluator(
    tasks,
    pre_dir,
    *,
    embedder,
    d_text,
    device,
    ctx_size_list,
    local_ctx_size,
    bfs_width,
    num_walks,
    walk_length,
    tokens_per_gpu,
    items_per_task,
    num_workers,
    context_seed,
    prefer_latest,
    shuffle_seed,
    mmap_populate,
    prefetch_factor,
    vector_db_path,
    db_cutoff,
    legacy_boolean=False,
    global_rank=0,
    local_rank=0,
    world_size=1,
    ddp=False,
):
    return Evaluator(
        tasks=tasks,
        pre_dir=pre_dir,
        eval_bs=max(1, tokens_per_gpu // max(ctx_size_list)),
        ctx_size_list=list(ctx_size_list),
        items_per_task=items_per_task,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
        local_ctx_size=local_ctx_size,
        bfs_width=bfs_width,
        num_walks=num_walks,
        walk_length=walk_length,
        prefer_latest=prefer_latest,
        mmap_populate=mmap_populate,
        legacy_boolean=legacy_boolean,
        embedder=embedder,
        d_text=d_text,
        shuffle_seed=shuffle_seed,
        context_seed=context_seed,
        vector_db_path=vector_db_path,
        db_cutoff=db_cutoff,
        global_rank=global_rank,
        local_rank=local_rank,
        world_size=world_size,
        ddp=ddp,
        device=device,
    )


def run_and_report(
    model, tasks, pre_dir, *, ctx_size, csv_out_dir, evaluator, embedder
):
    is_main = evaluator.global_rank == 0
    csv_out_dir = None if csv_out_dir is None else Path(csv_out_dir).expanduser()
    by_metric: dict[str, list[float]] = {}
    results = {}
    if is_main:
        log(eval_mode="plain", ctx_size=ctx_size)
    for task, _ctx, labels, preds_by_prefix, _nl, node_idxs in evaluator.evaluate_raw(
        [(model, "")], [ctx_size], with_node_idxs=True
    ):
        preds = preds_by_prefix[""]
        mname, mval, n, align, _ = _emit_and_score(
            csv_out_dir,
            task,
            pre_dir,
            embedder,
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
        log(
            indent=1,
            task=f"{task.db_name}/{task.table_name}",
            metric=mname,
            value=f"{mval:.4f}",
            n=n,
            align=align,
            norm_metric=nm,
            norm_value=f"{nv:.4f}",
        )
    if not is_main:
        return results
    for name, vals in by_metric.items():
        log(
            mean_metric=name,
            value=f"{sum(vals) / len(vals):.4f}",
            over_tasks=len(vals),
        )
    if csv_out_dir is not None:
        log(csv_dir=csv_out_dir)
    return results


def _is_better(task_type, a, b):
    return a > b if task_type == "clf" else a < b


def run_ensemble(
    model,
    pre_dir,
    val_tasks,
    test_tasks,
    *,
    grid,
    ctx_sizes,
    val_items,
    test_items,
    context_seed,
    val_ensemble_size,
    test_ensemble_size,
    tune_only,
    tuning_out_path,
    resume_path,
    csv_out_dir,
    targets,
    use_wandb,
    **eval_kwargs,
):
    embedder = eval_kwargs["embedder"]
    ddp = eval_kwargs.get("ddp", False)
    is_main = eval_kwargs.get("global_rank", 0) == 0

    resume_path = Path(resume_path).expanduser()
    guard = [
        sorted(grid),
        sorted(ctx_sizes),
        val_items,
        test_items,
        context_seed,
        val_ensemble_size,
        test_ensemble_size,
        tune_only,
        sorted(f"{t.db_name}/{t.table_name}" for t in (test_tasks or [])),
        eval_kwargs["shuffle_seed"],
        eval_kwargs["db_cutoff"],
    ]
    state = {}
    if resume_path.exists():
        state = torch.load(resume_path, weights_only=False)
        assert state["guard"] == guard, (
            f"{resume_path} was written by a different evaluation; "
            "delete it or use a fresh run_id"
        )
        log(
            resumed_from=str(resume_path),
            tuned=len((state.get("tune") or {"done": []})["done"]),
            seeds=(state.get("test") or {"seeds": 0})["seeds"],
        )

    def save_state(phase):
        if not is_main:
            return
        resume_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = resume_path.with_suffix(f".{os.getpid()}.tmp")
        try:
            torch.save({"guard": guard, **phase}, tmp)
            os.replace(tmp, resume_path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    cfgs = [(c, lcs, bw, pl) for lcs, bw, pl in grid for c in ctx_sizes if lcs <= c]
    if len(cfgs) == 1:
        assert val_tasks is None, "one configuration tunes nothing; pass val_tasks=None"
        best = {(t.db_name, t.table_name): {"cfg": cfgs[0]} for t in test_tasks}
    else:
        tune = state.get("tune") or {"best": {}, "scores": {}, "tuned": [], "done": []}
        best = tune["best"]
        scores = defaultdict(dict, tune["scores"])
        tuned = tune["tuned"]
        for lcs, bw, pl in grid:
            ctxs = [c for c in ctx_sizes if lcs <= c]
            if not ctxs:
                continue
            if [lcs, bw, pl] in tune["done"]:
                continue
            acc = {}
            for seed in range(val_ensemble_size):
                ev = build_evaluator(
                    val_tasks,
                    pre_dir,
                    items_per_task=val_items,
                    ctx_size_list=ctxs,
                    local_ctx_size=lcs,
                    bfs_width=bw,
                    prefer_latest=pl,
                    context_seed=member_context_seed(context_seed, seed),
                    **eval_kwargs,
                )
                for task, ctx, labels, preds_by_prefix, _nl in ev.evaluate_raw(
                    [(model, "")], ctxs
                ):
                    key = (task.db_name, task.table_name, ctx)
                    p = preds_by_prefix[""].astype(np.float64)
                    if key not in acc:
                        acc[key] = [labels, np.zeros_like(p), task.task_type]
                    acc[key][1] += p
            for (db, table, ctx), (labels, sp, task_type) in acc.items():
                key = (db, table)
                cfg = (ctx, lcs, bw, pl)
                _, v = metric_for(task_type, labels, sp / val_ensemble_size)
                if key not in best or _is_better(task_type, v, best[key]["value"]):
                    best[key] = {"cfg": cfg, "value": v, "task_type": task_type}
                scores[key][str(cfg)] = v
                log(
                    indent=1,
                    tune_task="/".join(key),
                    cfg=str(cfg).replace(" ", ""),
                    ens_size=val_ensemble_size,
                    value=f"{v:.4f}",
                )
                if use_wandb:
                    metric = METRIC_NAMES[task_type]
                    tuned.append([f"{db}/{table}", ctx, lcs, bw, pl, metric, v * 100])
                    wandb.log(
                        {
                            "tune/idx": len(tuned),
                            f"tune/{metric}/val/{db}/{table}": v * 100,
                            f"tune/best/{metric}/val/{db}/{table}": (
                                best[key]["value"] * 100
                            ),
                            "tune/ctx_size": ctx,
                            "tune/local_ctx_size": lcs,
                            "tune/bfs_width": bw,
                            "tune/prefer_latest": int(pl),
                            **{
                                f"target/tune/{k}": tv
                                for k, tv in targets.items()
                                if "/val/" in k and not k.endswith("/mean")
                            },
                        }
                    )
            tune["done"].append([lcs, bw, pl])
            state["tune"] = {**tune, "scores": dict(scores)}
            save_state({"tune": state["tune"]})

        if is_main:
            tuning_out_path = Path(tuning_out_path).expanduser()
            tuning_out_path.parent.mkdir(parents=True, exist_ok=True)
            tuning_out_path.write_text(
                json.dumps(
                    {
                        f"{db}/{table}": {
                            "best_cfg": list(best[db, table]["cfg"]),
                            "best_value": best[db, table]["value"],
                            "task_type": best[db, table]["task_type"],
                            "val_ensemble_size": val_ensemble_size,
                            "val_scores": by_cfg,
                        }
                        for (db, table), by_cfg in scores.items()
                    },
                    indent=2,
                )
            )
            log(tuning_written=str(tuning_out_path), tasks=len(scores))
        if use_wandb:
            wandb.log(
                {
                    "tune/scores": wandb.Table(
                        columns=["task", "ctx", "lcs", "bw", "pl", "metric", "value"],
                        data=tuned,
                    )
                }
            )
        if tune_only:
            return
        if ddp:
            payload = [best if is_main else None]
            dist.broadcast_object_list(payload, src=0)
            best = payload[0]

    groups = defaultdict(list)
    won_ctx = {}
    for t in test_tasks:
        b = best.get((t.db_name, t.table_name))
        if b is not None:
            ctx, lcs, bw, pl = b["cfg"]
            groups[lcs, bw, pl].append(t)
            won_ctx[t.db_name, t.table_name] = ctx

    csv_out_dir = None if csv_out_dir is None else Path(csv_out_dir).expanduser()
    test = state.get("test") or {
        "group": 0,
        "seeds": 0,
        "acc": {},
        "curve": {},
        "results": {},
    }
    curve: dict[int, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list),
        {k: defaultdict(list, v) for k, v in test["curve"].items()},
    )
    results = test["results"]
    if is_main:
        log(eval_mode="ensembled", configs=len(groups))
    assert not use_wandb or len(groups) == 1, (
        "wandb logging needs one context config across the run's tasks"
    )
    for gi, ((lcs, bw, pl), tasks) in enumerate(groups.items()):
        if gi < test["group"]:
            continue
        ctxs = sorted({won_ctx[t.db_name, t.table_name] for t in tasks})
        acc = {
            k: [
                labels,
                sp,
                next(t for t in tasks if (t.db_name, t.table_name) == k),
                ni,
            ]
            for k, (labels, sp, ni) in test["acc"].items()
        }
        for seed in range(test["seeds"], test_ensemble_size):
            ev = build_evaluator(
                tasks,
                pre_dir,
                items_per_task=test_items,
                ctx_size_list=ctxs,
                local_ctx_size=lcs,
                bfs_width=bw,
                prefer_latest=pl,
                context_seed=member_context_seed(context_seed, seed),
                **eval_kwargs,
            )
            for task, ctx, labels, preds_by_prefix, _nl, node_idxs in ev.evaluate_raw(
                [(model, "")], ctxs, with_node_idxs=True
            ):
                key = (task.db_name, task.table_name)
                if ctx != won_ctx[key]:
                    continue
                p = preds_by_prefix[""].astype(np.float64)
                if key not in acc:
                    acc[key] = [labels, np.zeros_like(p), task, node_idxs]
                acc[key][1] += p
            size = seed + 1
            full = size == test_ensemble_size
            point: dict[str, dict[str, float]] = defaultdict(dict)
            for key, (labels, sp, task, node_idxs) in acc.items():
                preds = sp / size
                mname, mval, n, align, _ = _emit_and_score(
                    csv_out_dir if full else None,
                    task,
                    pre_dir,
                    embedder,
                    labels,
                    preds,
                    node_idxs,
                )
                curve[size][mname].append(mval)
                if use_wandb:
                    _, nv = metric_for(task.task_type, labels, preds)
                    point[METRIC_NAMES[task.task_type]][
                        f"{task.split}/{task.db_name}/{task.table_name}"
                    ] = nv * 100.0
                if full:
                    results[f"{task.db_name}/{task.table_name}"] = {
                        "metric": mname,
                        "value": mval,
                        "cfg": (won_ctx[key], lcs, bw, pl),
                        "n": n,
                    }
                log(
                    indent=1,
                    task=f"{task.db_name}/{task.table_name}",
                    cfg=str((won_ctx[key], lcs, bw, pl)).replace(" ", ""),
                    ens_size=size,
                    metric=mname,
                    value=f"{mval:.4f}",
                    n=n,
                    align=align,
                )
            if use_wandb:
                logged = {"ens_size": size}
                for metric, per_task in point.items():
                    logged.update({f"{metric}/{k}": v for k, v in per_task.items()})
                    by_split = defaultdict(list)
                    for k, v in per_task.items():
                        by_split[k.split("/")[0]].append(v)
                    for split, vs in by_split.items():
                        logged[f"{metric}/{split}/mean"] = sum(vs) / len(vs)
                logged.update({f"target/{k}": v for k, v in targets.items()})
                wandb.log(logged)
            save_state(
                {
                    "tune": state.get("tune"),
                    "test": {
                        "group": gi,
                        "seeds": size,
                        "acc": {k: [v[0], v[1], v[3]] for k, v in acc.items()},
                        "curve": {k: dict(v) for k, v in curve.items()},
                        "results": results,
                    },
                }
            )
        test = {"group": gi + 1, "seeds": 0, "acc": {}, "curve": {}, "results": results}
    if not is_main:
        return results
    for size in sorted(curve):
        for name, vals in curve[size].items():
            log(
                ens_size=size,
                mean_metric=name,
                value=f"{sum(vals) / len(vals):.4f}",
                over_tasks=len(vals),
            )
    if csv_out_dir is not None:
        log(csv_dir=csv_out_dir)
    return results
