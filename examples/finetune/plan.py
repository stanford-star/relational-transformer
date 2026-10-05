import sys
from collections.abc import Sequence
from pathlib import Path

from examples.finetune import config
from examples.finetune.run import stage_dir
from reproduce.launch import Job, describe, run_sequential

PROJECT = "finetune"

MODELS: dict[str, str | None] = {
    "rt-j": "stanford-star/rt-j",
    "rt-plurel": "stanford-star/rt-plurel",
    "rt": None,
}

TASKS = (
    ("rel-hm", "item-sales"),
    ("rel-stack", "user-engagement"),
    ("rel-amazon", "user-churn"),
    ("rel-trial", "study-adverse"),
    ("rel-hm", "user-churn"),
    ("rel-amazon", "item-churn"),
    ("rel-event", "user-attendance"),
    ("rel-amazon", "user-ltv"),
    ("rel-avito", "user-visits"),
    ("rel-stack", "user-badge"),
    ("rel-event", "user-ignore"),
    ("rel-amazon", "item-ltv"),
    ("rel-trial", "site-success"),
    ("rel-stack", "post-votes"),
    ("rel-avito", "user-clicks"),
    ("rel-avito", "ad-ctr"),
    ("rel-trial", "study-outcome"),
    ("rel-event", "user-repeat"),
    ("rel-f1", "driver-dnf"),
    ("rel-f1", "driver-top3"),
    ("rel-f1", "driver-position"),
)


def prediction_table(
    out_root: str, project: str, model: str, db: str, task: str
) -> Path:
    return (
        stage_dir(out_root, project, f"{model}-{db}-{task}-test")
        / "eval_out"
        / f"{db}__{task}.csv"
    )


def jobs(
    *,
    models: Sequence[str],
    tasks: Sequence[tuple[str, str]],
    project: str,
    pre_dir: str,
    out_root: str,
    tokens_per_gpu: int,
    num_workers: int,
    eval_num_workers: int,
    selection_steps: int,
    patience_steps: int,
    eval_freq: int,
    eval_rows: int,
    selection_ensemble_size: int,
    tune_rows: int,
    test_ensemble_size: int,
    seed: int,
) -> list[Job]:
    assert models, "no models to plan; pass at least one of " + ", ".join(MODELS)
    out = []
    for model in models:
        assert model in MODELS, f"unknown model {model!r}; known: {', '.join(MODELS)}"
        for db, task in tasks:
            if prediction_table(out_root, project, model, db, task).exists():
                continue
            out.append(
                Job(
                    name=f"finetune-{model}-{db}-{task}",
                    target="examples.finetune.run:main",
                    args={
                        "model": model,
                        "db": db,
                        "task": task,
                        "load_ckpt_root": MODELS[model],
                        "pre_dir": pre_dir,
                        "tokens_per_gpu": tokens_per_gpu,
                        "num_workers": num_workers,
                        "eval_num_workers": eval_num_workers,
                        "selection_steps": selection_steps,
                        "patience_steps": patience_steps,
                        "eval_freq": eval_freq,
                        "eval_rows": eval_rows,
                        "selection_ensemble_size": selection_ensemble_size,
                        "tune_rows": tune_rows,
                        "test_ensemble_size": test_ensemble_size,
                        "seed": seed,
                        "project": project,
                        "out_root": out_root,
                    },
                )
            )
    return out


if __name__ == "__main__":
    plan = jobs(
        models=sys.argv[1:],
        tasks=TASKS,
        project=PROJECT,
        pre_dir=config.pre_dir(),
        out_root=config.out_root(),
        tokens_per_gpu=2**17,
        num_workers=8,
        eval_num_workers=2,
        selection_steps=50_000,
        patience_steps=10_000,
        eval_freq=100,
        eval_rows=1024,
        selection_ensemble_size=4,
        tune_rows=4096,
        test_ensemble_size=8,
        seed=0,
    )
    describe(plan)
    run_sequential(plan)
