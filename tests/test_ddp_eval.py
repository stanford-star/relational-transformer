import os
import socket
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import rt.eval._eval  # noqa: F401
from rt.eval.evaluator import Evaluator
from rt.model.net import SEM_TYPE_NAMES

CTX = 4
EVAL_BS = 2
N_BATCHES = 2
WORLD_SIZE = 2
PHANTOM = (1, 1, 1)


class _FakeTask:
    db_name = "rel-fake"
    table_name = "fake-task"
    split = "test"
    task_type = "clf"


def _fake_batch(rank, b):
    node_base = 100 * (rank + 1) + 10 * b
    is_targets = torch.zeros(EVAL_BS, CTX, dtype=torch.bool)
    is_targets[:, 0] = True
    node_idxs = torch.zeros(EVAL_BS, CTX, dtype=torch.int64)
    for i in range(EVAL_BS):
        node_idxs[i, 0] = node_base + i
        node_idxs[i, 1] = 900 + i
    col_name_idxs = torch.zeros(EVAL_BS, CTX, dtype=torch.int64)
    is_task_nodes = torch.ones(EVAL_BS, CTX, dtype=torch.bool)
    is_padding = torch.zeros(EVAL_BS, CTX, dtype=torch.bool)
    is_padding[:, 2:] = True
    sem_types = torch.full(
        (EVAL_BS, CTX), SEM_TYPE_NAMES.index("number"), dtype=torch.int64
    )
    number_values = torch.zeros(EVAL_BS, CTX, 1)
    for i in range(EVAL_BS):
        number_values[i, 0, 0] = float((node_base + i) % 2)
    batch_mask = torch.ones(EVAL_BS, dtype=torch.bool)
    for i in range(EVAL_BS):
        if (rank, b, i) == PHANTOM:
            batch_mask[i] = False
    return {
        "batch_mask": batch_mask,
        "is_targets": is_targets,
        "node_idxs": node_idxs,
        "col_name_idxs": col_name_idxs,
        "is_task_nodes": is_task_nodes,
        "is_padding": is_padding,
        "sem_types": sem_types,
        "number_values": number_values,
    }


class _FakeNet:
    def eval(self):
        return self

    def predict(self, batch, ctx_size_list, device, task):
        seed = (batch["node_idxs"] * batch["is_targets"].to(torch.int64)).sum(dim=1)
        return {c: seed.float() / 1000.0 for c in ctx_size_list}


def _make_evaluator(rank, ddp):
    ev = object.__new__(Evaluator)
    task = _FakeTask()
    ev.tasks = [task]
    ev.eval_splits = ["test"]
    ev.ctx_size_list = [CTX]
    ev.eval_bs = EVAL_BS
    ev.items_per_task = None
    ev.global_rank = rank
    ev.local_rank = rank
    ev.world_size = WORLD_SIZE if ddp else 1
    ev.ddp = ddp
    ev.device = "cpu"

    batches = [_fake_batch(rank, b) for b in range(N_BATCHES)]

    class _Loader:
        dataset = [None] * N_BATCHES

        def __iter__(self):
            return iter([{k: v.clone() for k, v in b.items()} for b in batches])

    loader = _Loader()
    ev.eval_loaders = {task: loader}
    ev.eval_loader_iters = {task: iter(loader)}
    return ev


def _expected(ddp):
    ranks = range(WORLD_SIZE) if ddp else [0]
    return [
        100 * (r + 1) + 10 * b + i
        for r in ranks
        for b in range(N_BATCHES)
        for i in range(EVAL_BS)
        if (r, b, i) != PHANTOM
    ]


def _run(rank, ddp, out):
    if ddp:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)
    ev = _make_evaluator(rank, ddp)
    yields = list(ev.evaluate_raw([(_FakeNet(), "")], [CTX], with_node_idxs=True))
    if rank == 0:
        assert len(yields) == 1, yields
        _task, ctx, labels, preds_by_prefix, num_labels, node_idxs = yields[0]
        out["ctx"] = ctx
        out["labels"] = labels.tolist()
        out["preds"] = preds_by_prefix[""].tolist()
        out["num_labels"] = num_labels.tolist()
        out["node_idxs"] = node_idxs.tolist()
    else:
        assert yields == []
        out["yields"] = 0
    if ddp:
        dist.destroy_process_group()


def _check(out, ddp):
    exp = _expected(ddp)
    assert out["node_idxs"] == exp
    assert out["ctx"] == CTX
    assert out["preds"] == pytest.approx([n / 1000.0 for n in exp])
    assert out["labels"] == pytest.approx([float(n % 2) for n in exp])
    assert out["num_labels"] == [1] * len(exp)


def test_eval_gather_single_process():
    out = {}
    _run(0, False, out)
    _check(out, ddp=False)


def _worker(rank, port, ret):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    out = {}
    _run(rank, True, out)
    ret[rank] = out


def test_eval_gather_ddp_two_ranks():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    ctx = mp.get_context("spawn")
    ret = ctx.Manager().dict()
    procs = [
        ctx.Process(target=_worker, args=(r, port, ret)) for r in range(WORLD_SIZE)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    for r, p in enumerate(procs):
        assert p.exitcode == 0, f"rank {r} exited {p.exitcode}"

    _check(dict(ret[0]), ddp=True)
    assert dict(ret[1]) == {"yields": 0}


class _EnsembleFakeEvaluator:
    def __init__(self, tasks, rank, ddp, value_for):
        self.tasks = tasks
        self.global_rank = rank
        self.ddp = ddp
        self.value_for = value_for

    def evaluate_raw(self, nets_with_prefix, ctx_list, with_node_idxs=False):
        for task in self.tasks:
            if self.ddp:
                dist.barrier()
            if self.global_rank != 0:
                continue
            labels = np.array([0.0, 1.0, 0.0, 1.0])
            p = self.value_for(task)
            preds = np.array([0.0, p, 0.0, p])
            out = (task, ctx_list[0], labels, {"": preds}, np.ones(4, dtype=np.int64))
            if with_node_idxs:
                out = out + (np.arange(4, dtype=np.int64),)
            yield out


def _run_ensemble_worker(rank, port, ret):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)

    m = sys.modules["rt.eval._eval"]

    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE)

    grid = [(4, 8, True), (4, 16, False)]
    val_tasks = [_named_task(f"t{i}", "val") for i in range(2)]
    test_tasks = [_named_task(f"t{i}", "test") for i in range(2)]

    def fake_build_evaluator(tasks, pre_dir, *, local_ctx_size, **kw):
        def value_for(task):
            good = local_ctx_size == (64 if task.table_name == "t0" else 128)
            return 1.0 if good else 0.5

        return _EnsembleFakeEvaluator(tasks, rank, True, value_for)

    def fake_emit_and_score(csv_dir, task, pre_dir, embedder, labels, preds, nidx):
        return ("auc", 1.0, len(labels), "ok", None)

    m.build_evaluator = fake_build_evaluator
    m._emit_and_score = fake_emit_and_score

    results = m.run_ensemble(
        object(),
        "unused-pre-dir",
        val_tasks,
        test_tasks,
        grid=grid,
        ctx_sizes=[CTX],
        val_items=None,
        test_items=None,
        val_ensemble_size=1,
        test_ensemble_size=1,
        tune_only=False,
        tuning_out_path=Path(tempfile.mkdtemp()) / "tuning.json",
        csv_out_dir=None,
        targets={},
        use_wandb=False,
        context_seed=0,
        shuffle_seed=0,
        db_cutoff=None,
        resume_path=Path(tempfile.mkdtemp()) / "eval-resume.json",
        embedder="test-embed",
        global_rank=rank,
        local_rank=rank,
        world_size=WORLD_SIZE,
        ddp=True,
    )
    ret[rank] = dict(results)
    dist.destroy_process_group()


def _named_task(table_name, split):
    t = _FakeTask()
    t.table_name = table_name
    t.split = split
    return t


def test_run_ensemble_ddp_does_not_deadlock():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    ctx = mp.get_context("spawn")
    ret = ctx.Manager().dict()
    procs = [
        ctx.Process(target=_run_ensemble_worker, args=(r, port, ret))
        for r in range(WORLD_SIZE)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    for r, p in enumerate(procs):
        assert p.exitcode == 0, f"rank {r} exited {p.exitcode} (None = hung)"

    assert set(dict(ret[0])) == {"rel-fake/t0", "rel-fake/t1"}
    assert dict(ret[1]) == {}


def test_eval_shards_are_disjoint_and_complete():
    num_items, bs = 7, 2
    world_size = 2
    n_batches = -(-num_items // (bs * world_size))
    seen = [
        rank * bs + idx * bs * world_size + i
        for rank in range(world_size)
        for idx in range(n_batches)
        for i in range(bs)
    ]
    real = sorted(x for x in seen if x < num_items)
    assert real == list(range(num_items))
    assert len(seen) == n_batches * bs * world_size
    assert np.all(np.diff(sorted(seen)) == 1)
