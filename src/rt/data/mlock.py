import contextlib
import ctypes
import ctypes.util
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from rt.data.tasks import resolve_db_task_list
from rt.progress import log

_libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
_libc.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_long,
]
_libc.mmap.restype = ctypes.c_void_p
_libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
_libc.mlock.restype = ctypes.c_int

_PROT_READ = 0x1
_MAP_SHARED = 0x01
_MAP_FAILED = ctypes.c_void_p(-1).value


def mlock_file(path: str) -> int:
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        if size == 0:
            raise RuntimeError(f"empty file: {path}")
        addr = _libc.mmap(None, size, _PROT_READ, _MAP_SHARED, fd, 0)
        if addr == _MAP_FAILED:
            err = ctypes.get_errno()
            raise OSError(err, f"mmap failed for {path}: {os.strerror(err)}")
    finally:
        os.close(fd)
    if _libc.mlock(addr, size) != 0:
        err = ctypes.get_errno()
        raise OSError(err, f"mlock failed for {path}: {os.strerror(err)}")
    return size


def mlock_main(
    *,
    db_task_list: list[tuple[str, str]] | str,
    pre_dir: str,
    embedder_ref: str,
    workers: int,
) -> None:
    db_names = sorted({db for db, _ in resolve_db_task_list(db_task_list)})
    log(mlock_dbs=len(db_names))

    def db_paths(db: str) -> list[str]:
        base = os.path.join(pre_dir, db)
        return [
            os.path.join(base, "nodes.rkyv"),
            os.path.join(base, f"text_emb_{embedder_ref}.bin"),
            os.path.join(base, "p2f_adj.rkyv"),
        ]

    def fmt_size(n: int) -> str:
        return f"{n / 2**30:.2f}GiB"

    page_size = os.sysconf("SC_PAGESIZE")

    def allocated_size(p: str) -> int:
        return os.stat(p).st_blocks * 512

    def footprint_size(p: str) -> int:
        size = os.stat(p).st_size
        return ((size + page_size - 1) // page_size) * page_size

    db_sizes: dict[str, int] = {}
    db_footprints: dict[str, int] = {}
    size_errors: dict[str, str] = {}
    for db in db_names:
        try:
            paths = db_paths(db)
            db_sizes[db] = sum(allocated_size(p) for p in paths)
            db_footprints[db] = sum(footprint_size(p) for p in paths)
        except Exception as e:  # noqa: BLE001
            size_errors[db] = f"{type(e).__name__}: {e}"

    total_size = sum(db_sizes.values())

    locked_files = 0
    total = 0
    skipped = 0

    for db in db_names:
        if db in size_errors:
            log(
                indent=1,
                db_size_error=db,
                error=size_errors[db].replace(" ", "_"),
            )
            skipped += 1

    def lock_db(db: str) -> tuple[str, int, Exception | None]:
        n = 0
        try:
            for p in db_paths(db):
                mlock_file(p)
                n += 1
        except Exception as e:  # noqa: BLE001
            return db, n, e
        return db, n, None

    pending = [db for db in db_names if db not in size_errors]
    total_footprint = sum(db_footprints[db] for db in pending)

    t0 = time.time()
    log(
        locking_dbs=len(pending),
        size=fmt_size(total_size),
        footprint=fmt_size(total_footprint),
    )
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(lock_db, db) for db in pending]
        for fut in as_completed(futures):
            db, n, err = fut.result()
            db_size = db_sizes[db]
            locked_files += n
            if err is not None:
                log(
                    indent=1,
                    db_lock_error=db,
                    size=fmt_size(db_size),
                    error=f"{type(err).__name__}:{err}".replace(" ", "_"),
                )
                skipped += 1
                continue
            log(indent=1, locked_db=db, size=fmt_size(db_size))
            total += db_size
    elapsed = time.time() - t0

    log(
        locked_files=locked_files,
        size=fmt_size(total),
        footprint=fmt_size(total_footprint),
        skipped_dbs=skipped,
        elapsed=f"{elapsed:.0f}s",
        rate=f"{total / 2**30 / max(elapsed, 1e-9):.2f}GiB/s",
        pid=os.getpid(),
    )
    log(sleeping_until_signaled=True)

    def _fast_exit(signum: int, frame: object) -> None:
        with contextlib.suppress(Exception):
            _libc.munlockall()
        os._exit(0)

    signal.signal(signal.SIGINT, _fast_exit)
    signal.signal(signal.SIGTERM, _fast_exit)
    signal.pause()
