from pathlib import Path

from reproduce import config
from reproduce.launch import Job, describe, run_sequential
from reproduce.tasks import reg_tasks, tasks

RT_CTX = [256, 512, 1024, 2048, 4096, 8192]
BASELINE_CTX = RT_CTX + [16384, 32768, 65536, 131072]
EXT_CTX = BASELINE_CTX[6:]
FULL = 10_000_000

METHOD_DEVICE = {
    "rt": "cuda",
    "rdblearn_tabicl": "cuda",
    "sql_tabicl": "cuda",
    "rdblearn_lgbm": "cpu",
    "sql_lgbm": "cpu",
}


def arms() -> dict[str, tuple[str, dict]]:
    share = config.share()
    out: dict[str, tuple[str, dict]] = {}
    for method in METHOD_DEVICE:
        out[f"fulltest/{method}"] = (
            method,
            dict(ctx_size_list=RT_CTX, items_per_task=FULL),
        )
    for method in ("rdblearn_tabicl", "sql_tabicl"):
        for ctx in EXT_CTX:
            out[f"fulltest_ext/{method}/{ctx}"] = (
                method,
                dict(ctx_size_list=[ctx], items_per_task=FULL),
            )
    out["subsampled/rt"] = ("rt", dict(ctx_size_list=RT_CTX, items_per_task=8192))
    for method in ("rdblearn_tabicl", "sql_tabicl", "rdblearn_lgbm", "sql_lgbm"):
        out[f"subsampled/{method}"] = (
            method,
            dict(ctx_size_list=BASELINE_CTX, items_per_task=8192),
        )
    out["abl/rand"] = (
        "rt",
        dict(
            ctx_size_list=RT_CTX, items_per_task=8192, num_walks=0, prefer_latest=False
        ),
    )
    out["abl/bfs32"] = (
        "rt",
        dict(
            ctx_size_list=RT_CTX, items_per_task=8192, local_ctx_size=8192, bfs_width=32
        ),
    )
    out["abl/bfs256"] = (
        "rt",
        dict(
            ctx_size_list=RT_CTX,
            items_per_task=8192,
            local_ctx_size=8192,
            bfs_width=256,
        ),
    )
    out["abl/vdb_rdblearn"] = (
        "rt",
        dict(
            ctx_size_list=RT_CTX,
            items_per_task=8192,
            vector_db_path=f"{share}/vector_db/rdblearn",
        ),
    )
    out["abl/vdb_rt"] = (
        "rt",
        dict(
            ctx_size_list=RT_CTX,
            items_per_task=8192,
            vector_db_path=f"{share}/vector_db/rt",
        ),
    )
    out["abl/nosem"] = (
        "rt",
        dict(
            ctx_size_list=RT_CTX,
            items_per_task=8192,
            pre_dir=f"{share}/relbench-preprocessed-nosem",
        ),
    )
    return out


def nosem_job() -> Job:
    return Job(
        name="nosem-data",
        target="reproduce.scaling.make_nosem_data:main",
        args=dict(
            pre_dir=config.pre_dir(),
            out_dir=f"{config.share()}/relbench-preprocessed-nosem",
            embedder="all-MiniLM-L12-v2",
            d_text=384,
            seed=0,
        ),
    )


def jobs(arm_names: list[str]) -> list[Job]:
    all_arms = arms()
    share, out_root = config.share(), config.out_root()
    out = []
    for arm in arm_names:
        method, overrides = all_arms[arm]
        pairs = reg_tasks() if arm.startswith("fulltest_ext/") else tasks()
        for db, table in pairs:
            out_dir = f"{out_root}/scaling/{arm}"
            if (Path(out_dir) / f"{db}__{table}.json").exists():
                continue
            out.append(
                Job(
                    name=f"scaling-{arm.replace('/', '-')}-{db}-{table}",
                    target="reproduce.scaling.run:main",
                    args=dict(
                        split="test",
                        pre_dir=config.pre_dir(),
                        features_root=f"{share}/features",
                        local_ctx_size=256,
                        bfs_width=32,
                        prefer_latest=True,
                        num_walks=10_000,
                        walk_length=20,
                        shuffle_seed=0,
                        context_seed=0,
                        tokens_per_gpu=2**18,
                        num_workers=8,
                        prefetch_factor=2,
                        mmap_populate=True,
                        db_cutoff=None,
                        vector_db_path=None,
                        ckpt=config.ckpt(),
                        tabicl_dir=f"{share}/tabicl",
                        tabicl_max_batch_size=1024,
                        tabicl_min_bin_size=48,
                        tabicl_softmax_temperature=0.9,
                        lgbm_n_jobs=24,
                    )
                    | overrides
                    | dict(
                        method=method,
                        device=METHOD_DEVICE[method],
                        db=db,
                        table=table,
                        out_dir=out_dir,
                    ),
                )
            )
    return out


if __name__ == "__main__":
    plan = [nosem_job()] + jobs(sorted(arms()))
    describe(plan)
    run_sequential(plan)
