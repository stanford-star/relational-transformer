from pathlib import Path

from pipelines.icl.models import MODELS
from reproduce import config
from reproduce.launch import Job, describe, run_sequential
from reproduce.tasks import tasks
from reproduce.tune.plan import load_configs

DEFAULT_CFG = (8192, 256, 32, True)
N_SEEDS = 16
ITEMS_PER_TASK = 8192
VARIANTS = ("default", "tuned")


def tuned_configs() -> dict:
    model = MODELS["rt-j"]
    return load_configs(model.configs_path, model.grid_stem, tasks())


def cfg(variant: str, db: str, table: str) -> tuple[int, int, int, bool]:
    if variant == "default":
        return DEFAULT_CFG
    ctx, lcs, bw, pl = tuned_configs()[f"{db}/{table}"]["best_cfg"]
    return int(ctx), int(lcs), int(bw), bool(pl)


def jobs(variants: tuple[str, ...] = VARIANTS) -> list[Job]:
    out_root = config.out_root()
    out = []
    for variant in variants:
        assert variant in VARIANTS, variant
        for db, table in tasks():
            ctx, lcs, bw, pl = cfg(variant, db, table)
            out_dir = f"{out_root}/enscurve/{variant}"
            if (Path(out_dir) / f"{db}__{table}.json").exists():
                continue
            out.append(
                Job(
                    name=f"enscurve-{variant}-{db}-{table}",
                    target="reproduce.enscurve.run:main",
                    args={
                        "variant": variant,
                        "device": "cuda",
                        "db": db,
                        "table": table,
                        "ctx_size": ctx,
                        "local_ctx_size": lcs,
                        "bfs_width": bw,
                        "prefer_latest": pl,
                        "n_seeds": N_SEEDS,
                        "items_per_task": ITEMS_PER_TASK,
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
