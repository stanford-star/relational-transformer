import sys

from examples.icl.models import model_from_argv
from examples.icl.tasks import task_list
from reproduce import config
from reproduce.launch import describe, run_sequential
from reproduce.leaderboard.plan import jobs

if __name__ == "__main__":
    model = model_from_argv(sys.argv)
    plan = jobs(
        ckpt=config.env(model.ckpt_env),
        configs_path=model.configs_path,
        grid_stem=model.grid_stem,
        task_list=task_list(),
        out_subdir=model.out_subdir,
    )
    describe(plan)
    run_sequential(plan)
