from reproduce.launch import Job
from rt.data import get_mixture_path
from rt.eval.legacy import run

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


def v1(*, pre_dir: str, out_dir: str) -> None:
    from rt.model.legacy.v1 import V1_HUB_REPO, V1Transformer

    def model_for_task(task):
        filename = f"pretrain_{task.db_name}_{task.table_name}.pt"
        print(f"loading {V1_HUB_REPO}/{filename}", flush=True)
        return V1Transformer.from_pretrained(filename, repo_id=V1_HUB_REPO)

    run(
        model_for_task,
        out_dir=out_dir,
        pre_dir=pre_dir,
        db_task_list=str(get_mixture_path("relbench", "forecast")),
        **CONTEXT,
    )


def plurel(*, pre_dir: str, out_dir: str, mode: str) -> None:
    from rt.model.legacy.plurel import (
        PLUREL_HUB_REPO,
        PLUREL_SYNTH_CKPT,
        PluRelTransformer,
    )

    assert mode in ("synth", "synth-real"), mode

    def model_for_task(task):
        filename = (
            PLUREL_SYNTH_CKPT
            if mode == "synth"
            else f"paper/cntd-pretrain_{task.db_name}_{task.table_name}.pt"
        )
        print(f"loading {PLUREL_HUB_REPO}/{filename}", flush=True)
        return PluRelTransformer.from_pretrained(filename, repo_id=PLUREL_HUB_REPO)

    run(
        model_for_task,
        out_dir=out_dir,
        pre_dir=pre_dir,
        db_task_list=str(get_mixture_path("relbench", "forecast")),
        **{**CONTEXT, "bfs_width": 128},
    )


def jobs(*, pre_dir: str, out_root: str) -> list[Job]:
    return [
        Job(
            name="eval-rt-v1",
            target="examples.eval.legacy:v1",
            args={"pre_dir": pre_dir, "out_dir": f"{out_root}/rt-v1"},
        ),
        Job(
            name="eval-rt-plurel-synth",
            target="examples.eval.legacy:plurel",
            args={
                "pre_dir": pre_dir,
                "out_dir": f"{out_root}/rt-plurel-synth",
                "mode": "synth",
            },
        ),
        Job(
            name="eval-rt-plurel-synth-real",
            target="examples.eval.legacy:plurel",
            args={
                "pre_dir": pre_dir,
                "out_dir": f"{out_root}/rt-plurel-synth-real",
                "mode": "synth-real",
            },
        ),
    ]


if __name__ == "__main__":
    from reproduce.launch import run_sequential

    run_sequential(
        jobs(pre_dir="data/relbench-preprocessed/legacy", out_root="~/ckpts/legacy")
    )
