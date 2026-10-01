from pathlib import Path

from reproduce import config
from reproduce.launch import Job, describe, run_sequential
from reproduce.tasks import tasks

CTX_GRID = [512, 1024, 2048, 4096, 8192]
LCS_BW_PL_GRID = [
    (lcs, bw, pl)
    for lcs in (256, 512, 1024, 2048, 4096, 8192)
    for bw in (8, 32, 128)
    for pl in (True, False)
]
VAL_ENSEMBLE_SIZE = 4
VAL_ITEMS_PER_TASK = 4096


def grid_path(db: str, table: str) -> Path:
    return (
        Path(config.out_root())
        / "no-entity"
        / "tune"
        / f"tune--{db}--{table}"
        / "tuning.json"
    )


def jobs() -> list[Job]:
    out_root = config.out_root()
    out = []
    for db, table in tasks():
        run_id = f"tune--{db}--{table}"
        if (grid_path(db, table)).exists():
            continue
        out.append(
            Job(
                name=f"tune-{db}-{table}",
                target="rt.eval:main",
                args=dict(
                    load_ckpt_path=config.ckpt(),
                    embedder="all-MiniLM-L12-v2",
                    d_text=384,
                    num_blocks=12,
                    d_model=512,
                    num_heads=8,
                    d_ff=2048,
                    splits=["val"],
                    db_task_list=[(db, table)],
                    pre_dir=config.pre_dir(),
                    tokens_per_gpu=2**18,
                    num_workers=8,
                    prefetch_factor=2,
                    num_walks=10_000,
                    walk_length=20,
                    val_items_per_task=VAL_ITEMS_PER_TASK,
                    test_items_per_task=None,
                    ctx_size_list=CTX_GRID,
                    mmap_populate=True,
                    shuffle_seed=0,
                    context_seed=0,
                    vector_db_path=None,
                    db_cutoff=None,
                    lcs_bw_pl_grid=LCS_BW_PL_GRID,
                    val_ensemble_size=VAL_ENSEMBLE_SIZE,
                    test_ensemble_size=1,
                    run_id=run_id,
                    run_name=run_id,
                    targets={},
                    project="tune",
                    entity=None,
                    out_root=out_root,
                    wandb_disabled=True,
                ),
            )
        )
    return out


if __name__ == "__main__":
    plan = jobs()
    describe(plan)
    run_sequential(plan)
