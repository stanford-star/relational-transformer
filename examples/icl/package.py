import sys
from pathlib import Path

from examples.icl.models import model_from_argv
from examples.icl.tasks import task_list
from reproduce import config
from reproduce.leaderboard.reduce import main

RESULTS_DIR = Path(__file__).parent / "results"

if __name__ == "__main__":
    model = model_from_argv(sys.argv)
    share = Path(config.share()) / model.out_subdir
    main(
        ckpt=config.env(model.ckpt_env),
        configs_path=model.configs_path,
        grid_stem=model.grid_stem,
        task_list=task_list(),
        out_subdir=model.out_subdir,
        csv_dir=str(share / "preds"),
        results_path=str(RESULTS_DIR / f"{model.name}.json"),
        zip_path=str(share / f"{model.zip_stem}.zip"),
    )
