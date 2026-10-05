from pathlib import Path

from examples.launch import Job
from examples.preprocess.run import embed_done, rustler_done


def databases(raw_dir: str) -> list[str]:
    raw = Path(raw_dir).expanduser()
    assert raw.is_dir(), (
        f"{raw} is not a directory; download the raw collection first "
        f"(see examples/README.md)"
    )
    names = sorted(p.parent.name for p in raw.glob("*/manifest.yaml"))
    assert names, f"{raw} holds no <database>/manifest.yaml"
    return names


def jobs(
    *,
    raw_dir: str,
    out_dir: str,
    source_repo: str,
    embedder: str,
    batch_size: int,
) -> list[Job]:
    out_root = Path(out_dir).expanduser()
    out = []
    for database in databases(raw_dir):
        pre_dataset_dir = out_root / database
        if not rustler_done(pre_dataset_dir):
            out.append(
                Job(
                    name=f"rustler/{database}",
                    target="examples.preprocess.run:rustler",
                    args={
                        "database": database,
                        "raw_dir": raw_dir,
                        "out_dir": out_dir,
                        "source": f"{source_repo}/{database}",
                    },
                )
            )
        if not embed_done(pre_dataset_dir, embedder):
            out.append(
                Job(
                    name=f"embed/{database}",
                    target="examples.preprocess.run:embed",
                    args={
                        "database": database,
                        "out_dir": out_dir,
                        "embedder": embedder,
                        "batch_size": batch_size,
                    },
                )
            )
    return out


def the_join_jobs() -> list[Job]:
    return jobs(
        raw_dir="data/the-join",
        out_dir="data/the-join-preprocessed",
        source_repo="stanford-star/the-join",
        embedder="all-MiniLM-L12-v2",
        batch_size=1024,
    )


def plurel_jobs() -> list[Job]:
    return jobs(
        raw_dir="data/plurel",
        out_dir="data/plurel-preprocessed",
        source_repo="stanford-star/plurel",
        embedder="all-MiniLM-L12-v2",
        batch_size=1024,
    )


def relbench_jobs() -> list[Job]:
    return jobs(
        raw_dir="data/relbench-v1",
        out_dir="data/relbench-preprocessed",
        source_repo="stanford-star/relbench-v1",
        embedder="all-MiniLM-L12-v2",
        batch_size=1024,
    )


if __name__ == "__main__":
    from examples.launch import run_sequential

    run_sequential(the_join_jobs())
