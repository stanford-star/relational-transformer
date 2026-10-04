import sys

from pipelines.icl.models import model_from_argv
from pipelines.icl.tasks import task_list
from reproduce.tune.collect import main

if __name__ == "__main__":
    model = model_from_argv(sys.argv)
    main(
        grid_stem=model.grid_stem,
        task_list=task_list(),
        configs_path=model.configs_path,
    )
