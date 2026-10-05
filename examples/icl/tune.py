import sys

from examples.icl.models import model_from_argv
from examples.icl.tasks import task_list
from reproduce import config
from reproduce.launch import describe, run_sequential
from reproduce.tune.plan import jobs

if __name__ == "__main__":
    model = model_from_argv(sys.argv)
    plan = jobs(
        ckpt=config.env(model.ckpt_env),
        grid_stem=model.grid_stem,
        task_list=task_list(),
    )
    describe(plan)
    run_sequential(plan)
