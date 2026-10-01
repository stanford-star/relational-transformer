import json
from pathlib import Path

import numpy as np

from reproduce import config
from reproduce.enscurve.plan import DEFAULT_CFG, ITEMS_PER_TASK, N_SEEDS, tuned_configs


def curve(variant: str, db: str, table: str) -> dict:
    path = Path(config.out_root()) / "enscurve" / variant / f"{db}__{table}.json"
    assert path.exists(), f"missing {path}; run reproduce.enscurve.plan first"
    return json.loads(path.read_text())


def main() -> None:
    cfgs = tuned_configs()
    assert len(cfgs) == 21, f"{len(cfgs)} tuned configs, expected 21"
    out = {}
    for task_key, rec in sorted(cfgs.items()):
        db, table = task_key.split("/")
        default = curve("default", db, table)
        tuned = curve("tuned", db, table)
        ctx, lcs, bw, pl = rec["best_cfg"]
        keys = ("ctx_size", "local_ctx_size", "bfs_width", "prefer_latest")
        assert [default["config"][k] for k in keys] == list(DEFAULT_CFG), (
            f"{task_key}: default curve config {default['config']}"
        )
        assert [tuned["config"][k] for k in keys] == [ctx, lcs, bw, bool(pl)], (
            f"{task_key}: tuned curve config {tuned['config']} != {rec['best_cfg']}"
        )
        for c in (default, tuned):
            assert (
                c["config"]["items_per_task"] == ITEMS_PER_TASK
                and c["config"]["n_seeds"] == N_SEEDS
            )
        out[task_key] = {
            "task_type": rec["task_type"],
            "tuned_cfg": rec["best_cfg"],
            "default": default["curve"]["1"],
            "tuned": tuned["curve"]["1"],
        }
        print(
            f"{task_key}: default={out[task_key]['default']:.4f} "
            f"tuned={out[task_key]['tuned']:.4f} "
            f"cfg=({ctx},{lcs},{bw},{'T' if pl else 'F'})",
            flush=True,
        )

    per_task = list(out.values())
    for tt in ("clf", "reg"):
        d = float(np.mean([r["default"] for r in per_task if r["task_type"] == tt]))
        t = float(np.mean([r["tuned"] for r in per_task if r["task_type"] == tt]))
        out[f"mean_{tt}"] = {"default": d, "tuned": t}
        print(f"mean {tt}: default={d:.4f} tuned={t:.4f}", flush=True)

    dest = Path(__file__).with_name("results.json")
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
