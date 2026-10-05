import json
import sys
from pathlib import Path

from examples import config
from examples.icl.models import model_from_argv
from examples.icl.tasks import task_list
from examples.launch import Job, describe, run_sequential

CTX_GRID = [512, 1024, 2048, 4096, 8192]
LCS_BW_PL_GRID = [
    (lcs, bw, pl)
    for lcs in (256, 512, 1024, 2048, 4096, 8192)
    for bw in (8, 32, 128)
    for pl in (True, False)
]
VAL_ENSEMBLE_SIZE = 4
VAL_ITEMS_PER_TASK = 4096


def grid_stem_prefix(grid_stem: str) -> str:
    prefix, _, rest = grid_stem.partition("{db}")
    assert prefix and rest.endswith("{table}"), (
        f"grid_stem {grid_stem!r} must look like 'tune-<model>--{{db}}--{{table}}'"
    )
    return prefix


def grid_path(grid_stem: str, db: str, table: str) -> Path:
    grid_stem_prefix(grid_stem)
    stem = grid_stem.format(db=db, table=table)
    return Path(config.out_root()) / "no-wandb-entity" / "tune" / stem / "tuning.json"


def load_configs(
    configs_path: str, grid_stem: str, task_list: list[tuple[str, str]]
) -> dict:
    cfgs = json.loads(Path(configs_path).read_text())
    want = {f"{db}/{table}" for db, table in task_list}
    assert set(cfgs) == want, (
        f"{configs_path} holds {len(cfgs)} tasks, the task list has {len(want)}; "
        f"only in the file: {sorted(set(cfgs) - want)}; "
        f"only in the task list: {sorted(want - set(cfgs))}"
    )
    prefix = grid_stem_prefix(grid_stem)
    for task_key, rec in sorted(cfgs.items()):
        assert rec["grid"].startswith(prefix), (
            f"{task_key}: tuned from {rec['grid']}, which is not a {prefix} grid. "
            f"{configs_path} was tuned for a different checkpoint, and context "
            f"configurations are only valid for the checkpoint they were tuned on."
        )
    return cfgs


def jobs(*, ckpt: str, grid_stem: str, task_list: list[tuple[str, str]]) -> list[Job]:
    out_root = config.out_root()
    out = []
    for db, table in task_list:
        if grid_path(grid_stem, db, table).exists():
            continue
        run_id = grid_stem.format(db=db, table=table)
        out.append(
            Job(
                name=f"tune-{db}-{table}",
                target="rt.eval:main",
                args={
                    "load_ckpt_path": ckpt,
                    "embedder": "all-MiniLM-L12-v2",
                    "d_text": 384,
                    "num_blocks": 12,
                    "d_model": 512,
                    "num_heads": 8,
                    "d_ff": 2048,
                    "splits": ["val"],
                    "db_task_list": [(db, table)],
                    "pre_dir": config.pre_dir(),
                    "tokens_per_gpu": 2**18,
                    "num_workers": 8,
                    "prefetch_factor": 2,
                    "num_walks": 10_000,
                    "walk_length": 20,
                    "val_items_per_task": VAL_ITEMS_PER_TASK,
                    "test_items_per_task": None,
                    "ctx_size_list": CTX_GRID,
                    "mmap_populate": True,
                    "shuffle_seed": 0,
                    "context_seed": 0,
                    "vector_db_path": None,
                    "db_cutoff": None,
                    "lcs_bw_pl_grid": LCS_BW_PL_GRID,
                    "val_ensemble_size": VAL_ENSEMBLE_SIZE,
                    "test_ensemble_size": 1,
                    "run_id": run_id,
                    "run_name": run_id,
                    "targets": {},
                    "project": "tune",
                    "wandb_entity": None,
                    "out_root": out_root,
                    "wandb_disabled": True,
                },
            )
        )
    return out


if __name__ == "__main__":
    model = model_from_argv(sys.argv)
    plan = jobs(
        ckpt=config.env(model.ckpt_env),
        grid_stem=model.grid_stem,
        task_list=task_list(),
    )
    describe(plan)
    run_sequential(plan)
