import json
import os
import re

import wandb

from reproduce import config

ENTITY = "rtv2"
MAE_KEY = "swa/nmae/val/mean"
AUC_KEY = "swa/auroc/val/mean"
MAX_STEP = 2**15

BASE_PROJECT = "2026-09-09_pretrain"
ABL_PROJECT = "2026-09-14-repaper-pretrain-abl"

ARMS = {
    "base": (BASE_PROJECT, "plurel-join"),
    "mask25": (ABL_PROJECT, "mask25-mw"),
    "mask50": (ABL_PROJECT, "mask50-mw"),
    "mask75": (ABL_PROJECT, "mask75-mw"),
    "mix-autocomplete": (ABL_PROJECT, "mix-autocomplete"),
    "mix-forecast": (ABL_PROJECT, "mix-forecast"),
}


def is_attempt(run_name: str, name: str) -> bool:
    if not run_name.startswith(name + "-"):
        return False
    return re.fullmatch(r"\d+(\.\d+)*", run_name[len(name) + 1 :]) is not None


def export_arm(api: wandb.Api, arm: str, project: str, name: str) -> None:
    runs = sorted(
        (
            r
            for r in api.runs(f"{ENTITY}/{project}")
            if r.name == name or is_attempt(r.name, name)
        ),
        key=lambda r: r.created_at,
    )
    assert runs, f"{arm}: no run named {name} in {ENTITY}/{project}"
    by_step: dict[int, tuple[float, float]] = {}
    for run in runs:
        for row in run.scan_history(keys=["step", MAE_KEY, AUC_KEY]):
            step, mae, auc = row.get("step"), row.get(MAE_KEY), row.get(AUC_KEY)
            if step is None or mae is None or auc is None or step > MAX_STEP:
                continue
            by_step[int(step)] = (float(mae), float(auc))
    steps = sorted(by_step)
    assert steps, f"{arm}: no validation rows under step {MAX_STEP}"
    out = {
        "arm": arm,
        "attempts": [r.name for r in runs],
        "max_step": MAX_STEP,
        "steps": steps,
        "mae": [by_step[s][0] for s in steps],
        "auc": [by_step[s][1] for s in steps],
    }
    dest = config.series_dir() / "pretrain_abl" / f"{arm}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"{arm}: {len(steps)} points, steps {steps[0]}..{steps[-1]} -> {dest}")


def main() -> None:
    assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY is not set"
    api = wandb.Api()
    for arm, (project, name) in ARMS.items():
        export_arm(api, arm, project, name)


if __name__ == "__main__":
    main()
