import json
import sys
from pathlib import Path

import numpy as np

from examples.icl import env
from examples.icl.ensemble import FULL, N_CFGS, N_SEEDS
from examples.icl.models import model_from_argv
from examples.icl.tasks import task_list
from examples.icl.tune import load_configs

RESULTS_DIR = Path(__file__).parent / "results"


def unit(out_subdir: str, db: str, table: str, rank: int) -> Path:
    return Path(env.out_root()) / out_subdir / f"cfg{rank}" / f"{db}__{table}"


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
    assert Path(cfg_rec["checkpoint"]).expanduser() == Path(ckpt).expanduser(), (
        f"{path}: produced from {cfg_rec['checkpoint']}, not {ckpt}"
    )
    assert cfg_rec["shuffle_seed"] == 0 and cfg_rec["context_seed"] == 0
    assert cfg_rec["db_cutoff"] is None
    assert cfg_rec["split"] == "test" and cfg_rec["items_per_task"] == FULL
    st = np.load(state_path)
    assert int(st["seeds"]) == N_SEEDS, f"{path}: {int(st['seeds'])}/{N_SEEDS} seeds"
    return st


def main(
    *,
    ckpt: str,
    configs_path: str,
    grid_stem: str,
    task_list: list[tuple[str, str]],
    out_subdir: str,
    csv_dir: str,
    results_path: str,
    zip_path: str,
) -> None:
    from rt.data import get_tasks
    from rt.eval.relbench import _emit_and_score

    pre_dir = env.pre_dir()
    csv_out = Path(csv_dir)
    cfgs = load_configs(configs_path, grid_stem, task_list)
    results = {}
    for task_key, rec in sorted(cfgs.items()):
        db, table = task_key.split("/")
        (task,) = get_tasks(pre_dir, [(db, table)], ("test",))
        total = labels = nodes = None
        for rank in range(N_CFGS):
            st = load_unit(
                unit(out_subdir, db, table, rank), rec["top_cfgs"][rank], ckpt
            )
            if total is None:
                total = st["sum_preds"].astype(np.float64)
                labels, nodes = st["labels"], st["node_idxs"]
            else:
                assert np.array_equal(nodes, st["node_idxs"])
                assert np.array_equal(labels, st["labels"])
                total = total + st["sum_preds"].astype(np.float64)
        mean_pred = total / (N_CFGS * N_SEEDS)

        mname, mval, n, align, _csv = _emit_and_score(
            csv_out, task, pre_dir, "all-MiniLM-L12-v2", labels, mean_pred, nodes
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
    dest = Path(results_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(
        f"\nmean clf: {summary['mean_clf']:.4f}  mean reg: {summary['mean_reg']:.4f}"
        f"\nwrote {dest}"
        f"\n\nprediction CSVs are under {csv_out}; validate and package them with"
        f"\n  python -m relbench.submit {csv_out} --out {zip_path}",
        flush=True,
    )


if __name__ == "__main__":
    model = model_from_argv(sys.argv)
    share = Path(env.share()) / model.out_subdir
    main(
        ckpt=env.env(model.ckpt_env),
        configs_path=model.configs_path,
        grid_stem=model.grid_stem,
        task_list=task_list(),
        out_subdir=model.out_subdir,
        csv_dir=str(share / "preds"),
        results_path=str(RESULTS_DIR / f"{model.name}.json"),
        zip_path=str(share / f"{model.zip_stem}.zip"),
    )
