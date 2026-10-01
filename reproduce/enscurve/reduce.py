import json
from pathlib import Path

import numpy as np

from reproduce import config
from reproduce.enscurve.plan import N_SEEDS, VARIANTS

N_TASKS = 21


def reduce_variant(variant: str) -> bool:
    root = Path(config.out_root()) / "enscurve" / variant
    paths = sorted(root.glob("*__*.json"))
    if len(paths) != N_TASKS:
        print(f"{variant}: {len(paths)}/{N_TASKS} task curves, not reduced", flush=True)
        return False
    recs = [json.loads(p.read_text()) for p in paths]
    rows = []
    for k in range(1, N_SEEDS + 1):
        row: dict = {"ens_size": k, "per_task": {}}
        by_type = {"clf": [], "reg": []}
        for r in recs:
            v = r["curve"][str(k)]
            by_type[r["task_type"]].append(v)
            row["per_task"][r["task"]] = v
        row["avg_auc"] = float(np.mean(by_type["clf"]))
        row["avg_mae"] = float(np.mean(by_type["reg"]))
        rows.append(row)
    out = {
        "variant": variant,
        "n_tasks": N_TASKS,
        "n_seeds": N_SEEDS,
        "task_types": {r["task"]: r["task_type"] for r in recs},
        "configs": {r["task"]: r["config"] for r in recs},
        "rows": rows,
    }
    dest = config.series_dir() / "enscurve" / f"{variant}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(
        f"wrote {dest} (ens=1 auc={rows[0]['avg_auc']:.4f} mae={rows[0]['avg_mae']:.4f}"
        f" -> ens={N_SEEDS} auc={rows[-1]['avg_auc']:.4f} "
        f"mae={rows[-1]['avg_mae']:.4f})",
        flush=True,
    )
    return True


def main() -> None:
    done = sum(reduce_variant(v) for v in VARIANTS)
    print(f"{done}/{len(VARIANTS)} variants reduced", flush=True)


if __name__ == "__main__":
    main()
