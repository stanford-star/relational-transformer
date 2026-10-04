import json
import time
from pathlib import Path

from rt.preprocess import embed_dataset, run_rustler_pre, update_meta_with_embeddings

RUSTLER_FILES = ("meta.json", "table_info.json", "column_index.json", "text.json")


def rustler_done(pre_dataset_dir: Path) -> bool:
    return all((pre_dataset_dir / f).exists() for f in RUSTLER_FILES)


def embed_done(pre_dataset_dir: Path, embedder: str) -> bool:
    meta_path = pre_dataset_dir / "meta.json"
    if not meta_path.exists():
        return False
    entry = json.loads(meta_path.read_text()).get("text_embeddings", {}).get(embedder)
    return bool(entry) and (pre_dataset_dir / entry["file"]).exists()


def rustler(*, database: str, raw_dir: str, out_dir: str, source: str) -> None:
    out_root = Path(out_dir).expanduser()
    raw = Path(raw_dir).expanduser() / database
    pre_dataset_dir = out_root / database

    if rustler_done(pre_dataset_dir):
        print(f"{database}: rustler already done", flush=True)
        return

    assert (raw / "manifest.yaml").is_file(), (
        f"{raw} has no manifest.yaml; a raw database is a RelBench-format "
        f"directory of parquet tables plus manifest.yaml"
    )

    started = time.monotonic()
    run_rustler_pre(raw, out_root, source=source, skip_tasks=False)
    print(f"{database}: rustler {time.monotonic() - started:.0f}s", flush=True)


def embed(*, database: str, out_dir: str, embedder: str, batch_size: int) -> None:
    pre_dataset_dir = Path(out_dir).expanduser() / database

    if embed_done(pre_dataset_dir, embedder):
        print(f"{database}: embeddings already done", flush=True)
        return

    assert rustler_done(pre_dataset_dir), (
        f"{pre_dataset_dir} has no rustler output to embed; run the rustler "
        f"stage for {database} first"
    )

    started = time.monotonic()
    d_text = embed_dataset(pre_dataset_dir, embedder, batch_size)
    update_meta_with_embeddings(pre_dataset_dir, embedder, d_text)
    print(
        f"{database}: embed {time.monotonic() - started:.0f}s  d_text {d_text}",
        flush=True,
    )
