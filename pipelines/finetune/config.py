import os
from pathlib import Path


def _env(name: str) -> str:
    value = os.environ.get(name)
    assert value, (
        f"{name} is not set. pipelines/finetune/README.md lists every variable "
        f"this package reads and what each one has to point at."
    )
    return str(Path(value).expanduser())


def pre_dir() -> str:
    return _env("RT_PRE_DIR")


def out_root() -> str:
    return _env("RT_OUT_ROOT")
