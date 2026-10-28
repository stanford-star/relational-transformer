import importlib
from importlib.metadata import PackageNotFoundError, version

RELBENCH_PIP = (
    '    pip install "relbench @ git+https://github.com/stanford-star/relbench@bac62c9"'
)


def import_relbench(purpose: str, module: str = "relbench"):
    try:
        found = version("relbench")
    except PackageNotFoundError:
        found = None
    if found is None or int(found.split(".")[0]) < 3:
        what = "is not installed" if found is None else f"is {found}, not the fork"
        raise ImportError(
            f"{purpose} needs the stanford-star fork of relbench (3.x), which rt "
            f"cannot declare as a dependency (git-only); relbench {what}. "
            f"Install it with:\n{RELBENCH_PIP}"
        )
    return importlib.import_module(module)
