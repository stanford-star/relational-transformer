import os
import shutil
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rt.progress import log

MARKER = ".staged"


def stage_paths(
    stage_dir: str,
    paths: list[str],
    *,
    local_rank: int,
    barrier: Callable[[], None],
) -> list[str]:
    root = Path(os.path.expandvars(stage_dir)).expanduser()
    srcs = [Path(p).expanduser() for p in paths]
    for s in srcs:
        assert s.is_dir(), f"nothing to stage at {s}"
    names = [s.name for s in srcs]
    assert len(set(names)) == len(names), f"staged paths share a basename: {names}"
    if local_rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        for src in srcs:
            dst = root / src.name
            assert os.stat(src).st_dev != os.stat(root).st_dev, (
                f"stage_dir {root} is on the same filesystem as {src}; "
                "staging is for a different (local) disk"
            )
            if (dst / MARKER).exists():
                log(staged=str(dst), reused=True)
                continue
            tic = time.time()
            if dst.exists():
                shutil.rmtree(dst)
            _copy(src, dst)
            (dst / MARKER).touch()
            size = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file())
            log(
                staged=str(dst),
                gb=f"{size / 1e9:.1f}",
                secs=f"{time.time() - tic:.0f}",
            )
    barrier()
    return [str(root / s.name) for s in srcs]


def _copy(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True)
    entries = sorted(src.iterdir())

    def one(e: Path) -> None:
        if e.is_dir():
            shutil.copytree(e, dst / e.name, symlinks=False)
        else:
            shutil.copy2(e, dst / e.name)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, entries))
