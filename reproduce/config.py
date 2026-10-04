import os
from pathlib import Path

from rt.data import get_mixture_path


def env(name: str) -> str:
    value = os.environ.get(name)
    assert value, (
        f"{name} is not set. The README of the stage you are running lists "
        f"every variable it reads and what each one has to point at."
    )
    return str(Path(value).expanduser())


def ckpt() -> str:
    return env("RT_CKPT")


def pre_dir() -> str:
    return env("RT_PRE_DIR")


def raw_dir() -> str:
    return env("RT_RAW_DIR")


def share() -> str:
    return env("RT_SHARE")


def out_root() -> str:
    return env("RT_OUT_ROOT")


def db_task_list() -> str:
    return str(get_mixture_path("relbench", "forecast"))


def series_dir() -> Path:
    return Path(__file__).parent / "series"
