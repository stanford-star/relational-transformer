import ast
import json
import sys
from pathlib import Path

from examples.icl.models import model_from_argv
from examples.icl.tasks import task_list
from examples.icl.tune import VAL_ENSEMBLE_SIZE, grid_path

N_CFGS = 120
N_TOP = 4


def main(
    *, grid_stem: str, task_list: list[tuple[str, str]], configs_path: str
) -> None:
    out = {}
    for db, table in task_list:
        path = grid_path(grid_stem, db, table)
        assert path.exists(), f"missing {path}; the grid is not finished"
        rec = json.loads(path.read_text())[f"{db}/{table}"]
        scores = rec["val_scores"]
        assert len(scores) == N_CFGS, (
            f"{db}/{table}: {len(scores)} configs scored, expected {N_CFGS}"
        )
        assert rec["val_ensemble_size"] == VAL_ENSEMBLE_SIZE, rec["val_ensemble_size"]
        reverse = rec["task_type"] == "clf"
        top = sorted(
            scores.items(), key=lambda kv: (-kv[1] if reverse else kv[1], kv[0])
        )[:N_TOP]
        assert list(ast.literal_eval(top[0][0])) == list(rec["best_cfg"]), (
            f"{db}/{table}: best_cfg {rec['best_cfg']} is not the top score {top[0]}"
        )
        out[f"{db}/{table}"] = {
            "task_type": rec["task_type"],
            "val_ensemble_size": rec["val_ensemble_size"],
            "best_cfg": rec["best_cfg"],
            "best_value": rec["best_value"],
            "top_cfgs": [list(ast.literal_eval(c)) for c, _ in top],
            "top_values": [v for _, v in top],
            "val_scores": scores,
            "grid": f"{grid_stem.format(db=db, table=table)}/tuning.json",
        }
    dest = Path(configs_path)
    dest.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(f"wrote {dest} ({len(out)} tasks)")
    for k, v in sorted(out.items()):
        print(f"  {k}: best={v['best_cfg']} ({v['best_value']:.4f})")


if __name__ == "__main__":
    model = model_from_argv(sys.argv)
    main(
        grid_stem=model.grid_stem,
        task_list=task_list(),
        configs_path=model.configs_path,
    )
