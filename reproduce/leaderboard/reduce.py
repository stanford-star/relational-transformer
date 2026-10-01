import json
from pathlib import Path

import numpy as np

from reproduce import config
from reproduce.enscurve.plan import tuned_configs
from reproduce.leaderboard.plan import N_CFGS, N_SEEDS


def unit(db: str, table: str, rank: int) -> Path:
    return Path(config.out_root()) / "leaderboard" / f"cfg{rank}" / f"{db}__{table}"


def load_unit(path: Path, cfg: list, ckpt: str) -> np.lib.npyio.NpzFile:
    result_path = path.with_suffix(".json")
    state_path = path.with_suffix(".state.npz")
    assert result_path.exists(), f"{path}: not finished"
    cfg_rec = json.loads(result_path.read_text())["config"]
    got = [
        cfg_rec["ctx_size"],
        cfg_rec["local_ctx_size"],
        cfg_rec["bfs_width"],
        cfg_rec["prefer_latest"],
    ]
    assert got == list(cfg), f"{path}: config {got} != tuned {cfg}"
    assert cfg_rec["n_seeds"] == N_SEEDS, f"{path}: {cfg_rec['n_seeds']} seeds"
    assert cfg_rec.get("checkpoint", ckpt) == ckpt, f"{path}: checkpoint {cfg_rec}"
    assert cfg_rec["shuffle_seed"] == 0 and cfg_rec["context_seed"] == 0
    assert cfg_rec["db_cutoff"] is None
    st = np.load(state_path)
    assert int(st["seeds"]) == N_SEEDS, f"{path}: {int(st['seeds'])}/{N_SEEDS} seeds"
    return st


def main() -> None:
    from rt.data import get_tasks
    from rt.eval.relbench import _emit_and_score

    pre_dir, ckpt = config.pre_dir(), config.ckpt()
    csv_dir = Path(config.share()) / "leaderboard" / "preds"
    cfgs = tuned_configs()
    assert len(cfgs) == 21, f"{len(cfgs)} tuned configs, expected 21"
    results = {}
    for task_key, rec in sorted(cfgs.items()):
        db, table = task_key.split("/")
        (task,) = get_tasks(pre_dir, [(db, table)], ("test",))
        total = labels = nodes = None
        for rank in range(N_CFGS):
            st = load_unit(unit(db, table, rank), rec["top_cfgs"][rank], ckpt)
            if total is None:
                total = st["sum_preds"].astype(np.float64)
                labels, nodes = st["labels"], st["node_idxs"]
            else:
                assert np.array_equal(nodes, st["node_idxs"])
                assert np.array_equal(labels, st["labels"])
                total = total + st["sum_preds"].astype(np.float64)
        mean_pred = total / (N_CFGS * N_SEEDS)

        mname, mval, n, align, _csv = _emit_and_score(
            csv_dir, task, pre_dir, "all-MiniLM-L12-v2", labels, mean_pred, nodes
        )
        results[task_key] = {
            "task_type": task.task_type,
            "metric": mname,
            "value": mval,
            "n": n,
            "align": align,
            "top_cfgs": rec["top_cfgs"],
            "top_values": rec["top_values"],
        }
        print(f"{task_key}: {mname}={mval:.4f} n={n} {align}", flush=True)

    by_type = {"clf": [], "reg": []}
    for r in results.values():
        by_type[r["task_type"]].append(r["value"])
    summary = {
        "n_cfgs": N_CFGS,
        "n_seeds": N_SEEDS,
        "mean_clf": float(np.mean(by_type["clf"])),
        "mean_reg": float(np.mean(by_type["reg"])),
        "per_task": results,
    }
    dest = config.series_dir() / "leaderboard" / "top4x4.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(
        f"\nmean clf: {summary['mean_clf']:.4f}  mean reg: {summary['mean_reg']:.4f}"
        f"\nwrote {dest}"
        f"\n\nprediction CSVs are under {csv_dir}; package them with"
        f"\n  python -m relbench.submit {csv_dir} --out {csv_dir.parent}/rt-j.zip",
        flush=True,
    )


if __name__ == "__main__":
    main()
