import os
from pathlib import Path


def _env(name: str) -> str:
    value = os.environ.get(name)
    assert value, (
        f"{name} is not set. reproduce/README.md lists every variable this "
        f"package reads and what each one has to point at."
    )
    return str(Path(value).expanduser())


def ckpt() -> str:
    return _env("RT_CKPT")


def pre_dir() -> str:
    return _env("RT_PRE_DIR")


def raw_dir() -> str:
    return _env("RT_RAW_DIR")


def share() -> str:
    return _env("RT_SHARE")


def out_root() -> str:
    return _env("RT_OUT_ROOT")


def db_task_list() -> str:
    return f"{pre_dir()}/db-task-lists/forecast.json"


def series_dir() -> Path:
    return Path(__file__).parent / "series"
