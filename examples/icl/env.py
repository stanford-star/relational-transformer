import os
from pathlib import Path


def env(name: str) -> str:
    value = os.environ.get(name)
    assert value, (
        f"{name} is not set. The README of the example you are running lists "
        f"every variable it reads and what each one has to point at."
    )
    return str(Path(value).expanduser())


def pre_dir() -> str:
    return env("RT_PRE_DIR")


def out_root() -> str:
    return env("RT_OUT_ROOT")


def share() -> str:
    return env("RT_SHARE")
