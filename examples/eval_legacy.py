from rt.eval.legacy import run
from rt.model.legacy.plurel import (
    PLUREL_HUB_REPO,
    PLUREL_SYNTH_CKPT,
    PluRelTransformer,
)
from rt.model.legacy.v1 import V1_HUB_REPO, V1Transformer

DATA_DIR = "data/relbench-preprocessed"
PRE_DIR = f"{DATA_DIR}/legacy"

CONTEXT = {
    "ctx_size": 1024,
    "local_ctx_size": 1024,
    "num_walks": 0,
    "walk_length": 0,
    "bfs_width": 256,
    "prefer_latest": False,
    "tokens_per_gpu": 2**18,
    "items_per_task": 10_000_000,
    "num_workers": 2,
    "prefetch_factor": 2,
    "shuffle_seed": 0,
    "context_seed": 0,
    "mmap_populate": True,
    "vector_db_path": None,
    "db_cutoff": None,
}


def eval_v1(out_dir: str = "eval_v1_out") -> None:
    def model_for_task(task):
        filename = f"pretrain_{task.db_name}_{task.table_name}.pt"
        print(f"loading {V1_HUB_REPO}/{filename}")
        return V1Transformer.from_pretrained(filename, repo_id=V1_HUB_REPO)

    run(
        model_for_task,
        out_dir=out_dir,
        pre_dir=PRE_DIR,
        db_task_list=f"{DATA_DIR}/db-task-lists/forecast.json",
        **CONTEXT,
    )


def eval_plurel(mode: str = "synth", out_dir: str = "eval_plurel_out") -> None:
    def model_for_task(task):
        filename = (
            PLUREL_SYNTH_CKPT
            if mode == "synth"
            else f"paper/cntd-pretrain_{task.db_name}_{task.table_name}.pt"
        )
        print(f"loading {PLUREL_HUB_REPO}/{filename}")
        return PluRelTransformer.from_pretrained(filename, repo_id=PLUREL_HUB_REPO)

    run(
        model_for_task,
        out_dir=out_dir,
        pre_dir=PRE_DIR,
        db_task_list=f"{DATA_DIR}/db-task-lists/forecast.json",
        **{**CONTEXT, "bfs_width": 128},
    )


if __name__ == "__main__":
    eval_v1()
