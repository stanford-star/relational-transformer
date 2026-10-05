import shutil
import sys
from pathlib import Path

from relbench.submit import evaluate_submission
from relbench.submit import main as package

from examples.finetune import config
from examples.finetune.plan import MODELS, PROJECT, TASKS, prediction_table


def gather(
    *,
    out_root: str,
    project: str,
    model: str,
    tasks: tuple[tuple[str, str], ...],
) -> tuple[Path, list[str]]:
    preds = Path(out_root).expanduser() / "leaderboard" / project / model
    preds.mkdir(parents=True, exist_ok=True)
    missing = []
    for db, task in tasks:
        src = prediction_table(out_root, project, model, db, task)
        if src.exists():
            shutil.copyfile(src, preds / src.name)
        else:
            missing.append(f"{db}/{task}")
    return preds, missing


def report(model: str, result: dict) -> None:
    print(f"\n{model}: test scores from relbench.submit")
    for key, entry in sorted(result["tasks"].items()):
        assert entry["status"] == "ok", f"{key}: {entry['error']}"
        print(f"  {key:28s} {entry['metric_name']:8s} {entry['metric']:.4f}")
    for family, rec in sorted(result["families"].items()):
        if not rec["present"]:
            continue
        print(
            f"  mean {family:16s} {rec['metric_name']:8s} {rec['aggregate']:.4f}"
            f"  ({rec['num_valid']}/{rec['num_total']} tasks)"
        )


def main(*, models: list[str], project: str, out_root: str) -> None:
    assert models, "no models to collect; pass at least one of " + ", ".join(MODELS)
    for model in models:
        assert model in MODELS, f"unknown model {model!r}; known: {', '.join(MODELS)}"
        preds, missing = gather(
            out_root=out_root, project=project, model=model, tasks=TASKS
        )
        print(
            f"== {model}: {len(TASKS) - len(missing)}/{len(TASKS)} prediction "
            f"tables in {preds}"
        )
        for m in missing:
            print(f"   missing {m}")
        assert not missing, (
            f"{model}: {len(missing)} tasks have no prediction table; run "
            f"examples.finetune.plan for them first"
        )
        base = preds.parent / f"{model}.zip"
        for family in ("classification", "regression"):
            zip_path = base.with_name(f"{model}-{family}.zip")
            assert not zip_path.exists(), f"{zip_path} exists; move it aside first"
        result = evaluate_submission(preds, verbose=False)
        report(model, result)
        assert package([str(preds), "--out", str(base)]) == 0, (
            f"{model}: relbench.submit validated no leaderboard family"
        )


if __name__ == "__main__":
    main(
        models=sys.argv[1:],
        project=PROJECT,
        out_root=config.out_root(),
    )
