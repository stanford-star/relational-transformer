from datetime import datetime

from examples.launch import Job


def job(
    *,
    name: str,
    ckpt: str,
    pre_dir: str,
    db_task_list: str | list[tuple[str, str]],
    out_root: str,
    project: str,
    run_id: str,
    tokens_per_gpu: int,
    num_workers: int,
    ctx_size_list: list[int],
    lcs_bw_pl_grid: list[tuple[int, int, bool]],
    test_ensemble_size: int,
) -> Job:
    return Job(
        name=name,
        target="rt.eval:main",
        args={
            "load_ckpt_path": ckpt,
            "embedder": "all-MiniLM-L12-v2",
            "d_text": 384,
            "num_blocks": 12,
            "d_model": 512,
            "num_heads": 8,
            "d_ff": 2048,
            "splits": ["test"],
            "db_task_list": db_task_list,
            "pre_dir": pre_dir,
            "tokens_per_gpu": tokens_per_gpu,
            "num_workers": num_workers,
            "prefetch_factor": 2,
            "num_walks": 10_000,
            "walk_length": 20,
            "val_items_per_task": None,
            "test_items_per_task": 10_000_000,
            "mmap_populate": True,
            "shuffle_seed": 0,
            "context_seed": 0,
            "vector_db_path": None,
            "db_cutoff": None,
            "ctx_size_list": ctx_size_list,
            "lcs_bw_pl_grid": lcs_bw_pl_grid,
            "val_ensemble_size": 1,
            "test_ensemble_size": test_ensemble_size,
            "run_id": run_id,
            "run_name": None,
            "targets": {},
            "project": project,
            "wandb_entity": None,
            "out_root": out_root,
            "wandb_disabled": True,
        },
    )


def rt_j_jobs() -> list[Job]:
    stamp = f"{datetime.now():%y-%m-%d_%H-%M-%S}"  # noqa: DTZ005
    return [
        job(
            name="eval-rt-j",
            ckpt="stanford-star/rt-j",
            pre_dir="data/relbench-preprocessed",
            db_task_list="data/db-task-lists/relbench-forecast.json",
            out_root="~/ckpts",
            project="rt-eval",
            run_id=stamp,
            tokens_per_gpu=2**18,
            num_workers=2,
            ctx_size_list=[8192],
            lcs_bw_pl_grid=[(256, 32, True)],
            test_ensemble_size=1,
        )
    ]


if __name__ == "__main__":
    from examples.launch import run_sequential

    run_sequential(rt_j_jobs())
