import sys
from pathlib import Path

from examples.icl import env
from examples.icl.models import model_from_argv
from examples.icl.tasks import task_list
from examples.icl.tune import load_configs
from examples.launch import Job, describe, run_sequential

N_CFGS = 4
N_SEEDS = 4
FULL = 10_000_000


def jobs(
    *,
    ckpt: str,
    configs_path: str,
    grid_stem: str,
    task_list: list[tuple[str, str]],
    out_subdir: str,
) -> list[Job]:
    out_root = env.out_root()
    cfgs = load_configs(configs_path, grid_stem, task_list)
    out = []
    for task_key, rec in sorted(cfgs.items()):
        db, table = task_key.split("/")
        for rank, (ctx, lcs, bw, pl) in enumerate(rec["top_cfgs"][:N_CFGS]):
            out_dir = f"{out_root}/{out_subdir}/cfg{rank}"
            if (Path(out_dir) / f"{db}__{table}.json").exists():
                continue
            out.append(
                Job(
                    name=f"leaderboard-cfg{rank}-{db}-{table}",
                    target="examples.icl.run:main",
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
                        "pre_dir": env.pre_dir(),
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
                        "ckpt": ckpt,
                    },
                )
            )
    return out


if __name__ == "__main__":
    model = model_from_argv(sys.argv)
    plan = jobs(
        ckpt=env.env(model.ckpt_env),
        configs_path=model.configs_path,
        grid_stem=model.grid_stem,
        task_list=task_list(),
        out_subdir=model.out_subdir,
    )
    describe(plan)
    run_sequential(plan)
