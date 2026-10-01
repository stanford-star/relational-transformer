from pathlib import Path

from reproduce import config
from reproduce.enscurve.plan import tuned_configs
from reproduce.launch import Job, describe, run_sequential

N_CFGS = 4
N_SEEDS = 4
FULL = 10_000_000


def jobs() -> list[Job]:
    out_root = config.out_root()
    out = []
    for task_key, rec in sorted(tuned_configs().items()):
        db, table = task_key.split("/")
        for rank, (ctx, lcs, bw, pl) in enumerate(rec["top_cfgs"][:N_CFGS]):
            out_dir = f"{out_root}/leaderboard/cfg{rank}"
            if (Path(out_dir) / f"{db}__{table}.json").exists():
                continue
            out.append(
                Job(
                    name=f"leaderboard-cfg{rank}-{db}-{table}",
                    target="reproduce.enscurve.run:main",
                    args={
                        "variant": f"cfg{rank}",
                        "device": "cuda",
                        "db": db,
                        "table": table,
                        "ctx_size": int(ctx),
                        "local_ctx_size": int(lcs),
                        "bfs_width": int(bw),
                        "prefer_latest": bool(pl),
                        "n_seeds": N_SEEDS,
                        "items_per_task": FULL,
                        "split": "test",
                        "pre_dir": config.pre_dir(),
                        "out_dir": out_dir,
                        "num_walks": 10_000,
                        "walk_length": 20,
                        "shuffle_seed": 0,
                        "context_seed": 0,
                        "tokens_per_gpu": 2**18,
                        "num_workers": 8,
                        "prefetch_factor": 2,
                        "mmap_populate": True,
                        "db_cutoff": None,
                        "ckpt": config.ckpt(),
                    },
                )
            )
    return out


if __name__ == "__main__":
    plan = jobs()
    describe(plan)
    run_sequential(plan)
