import shutil
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi

from rt.preprocess._preprocess import (
    dataset_name,
    embed_dataset,
    resolve_dataset_dir,
    run_rustler_pre,
    update_meta_with_embeddings,
)

CAST_TO_BOOL: dict[tuple[str, str], list[str]] = {
    ("rel-amazon", "user-churn"): ["churn"],
    ("rel-amazon", "item-churn"): ["churn"],
    ("rel-stack", "user-engagement"): ["contribution"],
    ("rel-stack", "user-badge"): ["WillGetBadge"],
    ("rel-trial", "study-outcome"): ["outcome"],
    ("rel-f1", "driver-dnf"): ["did_not_finish"],
    ("rel-f1", "driver-top3"): ["qualifying"],
    ("rel-hm", "user-churn"): ["churn"],
    ("rel-event", "user-repeat"): ["target"],
    ("rel-event", "user-ignore"): ["target"],
    ("rel-avito", "user-visits"): ["num_click"],
    ("rel-avito", "user-clicks"): ["num_click"],
}

BINARIZE_FIRST: dict[tuple[str, str], list[str]] = {
    ("rel-stack", "postLinks"): ["LinkTypeId"],
    ("rel-trial", "studies"): ["has_dmc"],
    ("rel-trial", "eligibilities"): ["adult", "child", "older_adult"],
}


def _transform_df(df, db_name: str, table_name: str):
    for col in CAST_TO_BOOL.get((db_name, table_name), []):
        if col in df.columns:
            df = df.with_columns(pl.col(col).cast(pl.Boolean).alias(col))

    for col in BINARIZE_FIRST.get((db_name, table_name), []):
        if col in df.columns:
            s = df[col].cast(pl.String)
            first = s.drop_nulls().first()
            df = df.with_columns(
                pl.when(pl.col(col).is_null())
                .then(None)
                .otherwise(s == first)
                .alias(col)
            )

    if db_name == "rel-amazon" and table_name == "product" and "category" in df.columns:
        df = df.with_columns(
            pl.col("category").list.first().cast(pl.String).alias("category")
        )

    return df


def transform_dataset(dataset_dir: Path, out_dataset_dir: Path, db_name: str) -> Path:
    out_dataset_dir = Path(out_dataset_dir)
    if out_dataset_dir.exists():
        shutil.rmtree(out_dataset_dir)
    out_dataset_dir.mkdir(parents=True)

    for src in sorted(Path(dataset_dir).rglob("*")):
        rel = src.relative_to(dataset_dir)
        dst = out_dataset_dir / rel
        if src.is_dir():
            dst.mkdir(parents=True, exist_ok=True)
            continue
        if src.suffix == ".parquet":
            parts = rel.parts
            table_name = Path(parts[-1]).stem if parts[0] == "db" else parts[1]
            df = pl.read_parquet(src)
            out = _transform_df(df, db_name, table_name)
            out.write_parquet(dst)
            changed = [
                c
                for c in out.columns
                if c in df.columns and out.schema[c] != df.schema[c]
            ]
            if changed:
                print(
                    f"  {rel}: {', '.join(f'{c}->{out.schema[c]}' for c in changed)}",
                    flush=True,
                )
        else:
            shutil.copy2(src, dst)
    return out_dataset_dir


def rustler_one_legacy(
    spec: str,
    out_dir: Path,
    *,
    revision: str | None = None,
) -> Path:
    out_dir = Path(out_dir).expanduser()
    dataset_dir = resolve_dataset_dir(spec, revision=revision)
    name = dataset_name(dataset_dir)

    tf_dir = out_dir / "_transformed" / name
    print(f"=== legacy-transforming {name} ({spec}) -> {tf_dir} ===", flush=True)
    transform_dataset(dataset_dir, tf_dir, name)

    pre_dataset_dir = out_dir / name
    print(f"=== preprocessing {name} -> {pre_dataset_dir} ===", flush=True)
    run_rustler_pre(tf_dir, out_dir, source=spec, skip_tasks=False)
    return pre_dataset_dir


def preprocess_one_legacy(
    spec: str,
    out_dir: Path,
    *,
    embedder: str,
    batch_size: int,
    upload_repo: str | None,
    private: bool,
    revision: str | None,
) -> Path:
    out_dir = Path(out_dir).expanduser()
    pre_dataset_dir = rustler_one_legacy(spec, out_dir, revision=revision)
    name = pre_dataset_dir.name
    d_text = embed_dataset(pre_dataset_dir, embedder, batch_size)
    update_meta_with_embeddings(pre_dataset_dir, embedder, d_text)

    if upload_repo:
        api = HfApi()
        api.create_repo(
            upload_repo, repo_type="dataset", private=private, exist_ok=True
        )
        print(f"uploading {pre_dataset_dir} -> {upload_repo}/legacy/{name}", flush=True)
        api.upload_folder(
            folder_path=str(pre_dataset_dir),
            path_in_repo=f"legacy/{name}",
            repo_id=upload_repo,
            repo_type="dataset",
            commit_message=f"add legacy (RT-v1 boolean typing) preprocessed {name}",
        )
        print(f"uploaded {upload_repo}/legacy/{name}", flush=True)
    return pre_dataset_dir
