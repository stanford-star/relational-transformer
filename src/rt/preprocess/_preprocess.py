import json
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi, snapshot_download

from rt.preprocess.embed import embed_texts
from rt.rustler import preprocess


def resolve_repo(spec: str) -> tuple[str, str]:
    parts = spec.strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(
            f"{spec!r} is not a Hub 'org/name[/subdir]' spec or a local path."
        )
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def resolve_dataset_dir(spec: str, revision: str | None = None) -> Path:
    p = Path(spec).expanduser()
    if (p / "manifest.yaml").exists():
        return p
    repo_id, subdir = resolve_repo(spec)
    if not subdir:
        return Path(
            snapshot_download(repo_id=repo_id, revision=revision, repo_type="dataset")
        )
    local_root = snapshot_download(
        repo_id=repo_id,
        revision=revision,
        repo_type="dataset",
        allow_patterns=f"{subdir}/*",
    )
    return Path(local_root) / subdir


def dataset_name(dataset_dir: Path) -> str:
    text = (dataset_dir / "manifest.yaml").read_text()
    try:
        return yaml.safe_load(text)["name"]
    except Exception:  # noqa: BLE001
        for line in text.splitlines():
            if line.startswith("name:"):
                return line.split(":", 1)[1].strip().strip("'\"")
    raise ValueError(f"no 'name' in {dataset_dir / 'manifest.yaml'}")


def run_rustler_pre(
    dataset_dir: Path, out_dir: Path, source: str, skip_tasks: bool
) -> None:
    print(f"+ preprocess {dataset_dir} -> {out_dir}", flush=True)
    preprocess(str(dataset_dir), str(out_dir), source=source, skip_tasks=skip_tasks)


def embed_dataset(pre_dataset_dir: Path, embedder: str, batch_size: int) -> int:
    out_root = pre_dataset_dir.parent
    name = pre_dataset_dir.name
    embed_texts(
        dataset_name=name,
        pre_dir=str(out_root),
        device=None,
        batch_size=batch_size,
        embedder=embedder,
    )
    emb_path = pre_dataset_dir / f"text_emb_{embedder}.bin"
    num_text = len(json.loads((pre_dataset_dir / "text.json").read_text()))
    d_text = emb_path.stat().st_size // (max(num_text, 1) * 2)
    return d_text


def _embeddings_done(pre_dataset_dir: Path) -> bool:
    meta_path = pre_dataset_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        embs = json.loads(meta_path.read_text()).get("text_embeddings", {})
    except Exception:  # noqa: BLE001
        return False
    return bool(embs) and all(
        (pre_dataset_dir / e["file"]).exists() for e in embs.values()
    )


def update_meta_with_embeddings(
    pre_dataset_dir: Path, embedder: str, d_text: int
) -> None:
    meta_path = pre_dataset_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta.setdefault("text_embeddings", {})[embedder] = {
        "file": f"text_emb_{embedder}.bin",
        "d_text": d_text,
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")


def meta_without_text(pre_dataset_dir: Path) -> dict:
    meta = json.loads((pre_dataset_dir / "meta.json").read_text())
    meta["files"].pop("text", None)
    return meta


def upload_dataset(
    pre_dataset_dir: Path, repo: str, private: bool, retain_text: bool
) -> None:
    name = pre_dataset_dir.name
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    print(f"uploading {pre_dataset_dir} -> {repo}/{name}", flush=True)
    api.upload_folder(
        folder_path=str(pre_dataset_dir),
        path_in_repo=name,
        repo_id=repo,
        repo_type="dataset",
        commit_message=f"add preprocessed {name}",
        ignore_patterns=None if retain_text else ["text.json", "meta.json"],
    )
    if not retain_text:
        api.upload_file(
            path_or_fileobj=json.dumps(
                meta_without_text(pre_dataset_dir), indent=2
            ).encode()
            + b"\n",
            path_in_repo=f"{name}/meta.json",
            repo_id=repo,
            repo_type="dataset",
            commit_message=f"add preprocessed {name}: meta.json without text.json",
        )
    print(f"uploaded {repo}/{name}", flush=True)


def bulk_upload(out_dir: Path, repo: str, private: bool) -> None:
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", private=private, exist_ok=True)
    print(f"bulk-uploading {out_dir} -> {repo} (upload_large_folder)", flush=True)
    api.upload_large_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(out_dir),
    )
    print(f"bulk-uploaded {out_dir} -> {repo}", flush=True)


def preprocess_one(
    spec: str,
    out_dir: Path,
    *,
    embedder: str,
    batch_size: int,
    skip_tasks: bool,
    embed: bool = True,
    upload_repo: str | None,
    private: bool,
    retain_text: bool,
    revision: str | None,
) -> Path:
    dataset_dir = resolve_dataset_dir(spec, revision=revision)
    name = dataset_name(dataset_dir)
    pre_dataset_dir = out_dir / name
    print(f"=== preprocessing {name} ({spec}) -> {pre_dataset_dir} ===", flush=True)

    run_rustler_pre(dataset_dir, out_dir, source=spec, skip_tasks=skip_tasks)
    if embed:
        d_text = embed_dataset(pre_dataset_dir, embedder, batch_size)
        update_meta_with_embeddings(pre_dataset_dir, embedder, d_text)
    if upload_repo:
        upload_dataset(
            pre_dataset_dir, upload_repo, private=private, retain_text=retain_text
        )
    return pre_dataset_dir


def list_datasets(repo: str, revision: str | None = None) -> list[str]:
    api = HfApi()
    files = api.list_repo_files(repo, repo_type="dataset", revision=revision)
    subdirs = sorted(
        {
            f.split("/", 1)[0]
            for f in files
            if f.endswith("/manifest.yaml") and f.count("/") == 1
        }
    )
    return [f"{repo}/{d}" for d in subdirs]


def one(
    *,
    dataset: str,
    out_dir: str,
    embedder: str,
    batch_size: int,
    skip_tasks: bool,
    embed: bool,
    upload_repo: str | None,
    public: bool,
    retain_text: bool,
    revision: str | None,
) -> None:
    preprocess_one(
        dataset,
        Path(out_dir).expanduser(),
        embedder=embedder,
        batch_size=batch_size,
        skip_tasks=skip_tasks,
        embed=embed,
        upload_repo=upload_repo,
        private=not public,
        retain_text=retain_text,
        revision=revision,
    )


def many(
    *,
    repo: str,
    out_dir: str,
    shard: int,
    num_shards: int,
    skip_existing: bool,
    embedder: str,
    batch_size: int,
    skip_tasks: bool,
    embed: bool,
    upload_repo: str | None,
    public: bool,
    retain_text: bool,
    revision: str | None,
) -> None:
    specs = list_datasets(repo, revision=revision)
    assert 0 <= shard < num_shards, (
        f"shard must be in [0, num_shards); got shard={shard} num_shards={num_shards}"
    )
    shard = specs[shard::num_shards]
    print(
        f"shard {shard}/{num_shards}: {len(shard)} of {len(specs)} datasets",
        flush=True,
    )
    out_dir = Path(out_dir).expanduser()
    failures = []
    for i, spec in enumerate(shard):
        name = spec.rsplit("/", 1)[-1]
        if skip_existing and _embeddings_done(out_dir / name):
            print(f"[{i + 1}/{len(shard)}] skip existing {name}", flush=True)
            continue
        print(f"[{i + 1}/{len(shard)}] {spec}", flush=True)
        try:
            preprocess_one(
                spec,
                out_dir,
                embedder=embedder,
                batch_size=batch_size,
                skip_tasks=skip_tasks,
                embed=embed,
                upload_repo=upload_repo,
                private=not public,
                retain_text=retain_text,
                revision=revision,
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"  FAILED {spec}: {type(e).__name__}: {e}", file=sys.stderr, flush=True
            )
            failures.append(spec)
    if failures:
        print(f"\n{len(failures)} failure(s):", file=sys.stderr)
        for s in failures:
            print(f"  {s}", file=sys.stderr)
        sys.exit(1)


def ls(*, repo: str, revision: str | None) -> None:
    for spec in list_datasets(repo, revision=revision):
        print(spec)


def upload(
    *, pre_dir: str, repo: str, bulk: bool, public: bool, retain_text: bool
) -> None:
    path = Path(pre_dir).expanduser()
    if bulk:
        assert retain_text, (
            "bulk_upload uploads the folder as it is on disk; it cannot drop "
            "text.json, so pass retain_text=True or upload per dataset"
        )
        bulk_upload(path, repo, private=not public)
    else:
        upload_dataset(path, repo, private=not public, retain_text=retain_text)
