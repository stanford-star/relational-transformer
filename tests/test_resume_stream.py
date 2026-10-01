from __future__ import annotations

import random

import pytest

from rt.data.datasets import resume_positions

CFG = {
    "seed": 12345,
    "train_tokens_per_gpu": 2**17,
    "total_bs": 256,
    "world_size": 1,
}


def _grad_accum(ctx: int) -> int:
    bs = max(1, CFG["train_tokens_per_gpu"] // ctx)
    if CFG["total_bs"] < CFG["world_size"] * bs:
        return 1
    return CFG["total_bs"] // (CFG["world_size"] * bs)


def _ctx_at(ctx_list: list[int], t: int) -> int:
    if len(ctx_list) == 1:
        return ctx_list[0]
    return random.Random(CFG["seed"] + t).choice(ctx_list)


def _next_index(target: int, worker_id: int, stride: int) -> int:
    i = worker_id
    while i < target:
        i += stride
    return i


def _walk(ctx_list, start_step, worker_id, stride):
    if len(ctx_list) == 1:
        consumed = start_step * _grad_accum(ctx_list[0])
        nxt = _next_index(consumed, worker_id, stride)
        return nxt, nxt
    ctx_step = _next_index(start_step, worker_id, stride)
    mine = [t for t in range(worker_id, start_step, stride)]
    sampler_step = worker_id + stride * sum(
        _grad_accum(_ctx_at(ctx_list, t)) for t in mine
    )
    return ctx_step, sampler_step


@pytest.mark.parametrize("ctx_list", [[128, 256, 512, 1024], [1024]])
@pytest.mark.parametrize("stride", [1, 2, 8])
@pytest.mark.parametrize("start_step", [0, 1, 7, 100, 2300, 12_345])
def test__resume_positions__match_an_uninterrupted_walk(
    ctx_list: list[int], stride: int, start_step: int
) -> None:
    for worker_id in range(stride):
        assert resume_positions(
            train_ctx_size_list=ctx_list,
            start_step=start_step,
            worker_id=worker_id,
            stride=stride,
            **CFG,
        ) == _walk(ctx_list, start_step, worker_id, stride)


def test__resume_at_zero__is_a_fresh_run() -> None:
    for worker_id in (0, 1, 5):
        assert resume_positions(
            train_ctx_size_list=[128, 256, 512, 1024],
            start_step=0,
            worker_id=worker_id,
            stride=8,
            **CFG,
        ) == (worker_id, worker_id)


def test__resume_is_a_continuation__not_a_replay_and_not_a_jump() -> None:
    ctx_list = [128, 256, 512, 1024]
    for stride in (1, 4):
        for worker_id in range(stride):
            a = resume_positions(
                train_ctx_size_list=ctx_list,
                start_step=500,
                worker_id=worker_id,
                stride=stride,
                **CFG,
            )
            whole = resume_positions(
                train_ctx_size_list=ctx_list,
                start_step=1500,
                worker_id=worker_id,
                stride=stride,
                **CFG,
            )
            ctx_step, sampler_step = a
            while ctx_step < 1500:
                ctx = random.Random(CFG["seed"] + ctx_step).choice(ctx_list)
                bs = max(1, CFG["train_tokens_per_gpu"] // ctx)
                ga = (
                    1
                    if CFG["total_bs"] < CFG["world_size"] * bs
                    else CFG["total_bs"] // (CFG["world_size"] * bs)
                )
                sampler_step += stride * ga
                ctx_step += stride
            assert (ctx_step, sampler_step) == whole


def test__a_step__is_never_evaluated_twice() -> None:
    import inspect

    import rt.train._train as t

    src = inspect.getsource(t.main)
    assert "if eval_freq and step % eval_freq == 0 and step != evaled_at:" in src, (
        "the eval trigger must skip a step already evaluated"
    )
    assert "eval_freq <= total_steps" in src
