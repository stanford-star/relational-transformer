import json
from functools import cache
from pathlib import Path

CORE_FILES = (
    "meta.json",
    "nodes.rkyv",
    "offsets.rkyv",
    "p2f_adj.rkyv",
    "table_info.json",
    "column_index.json",
)

METADATA_FILES = ("meta.json", "table_info.json", "column_index.json")


def resolve_repo(spec: str) -> tuple[str, str]:
    parts = str(spec).strip("/").split("/")
    if len(parts) < 2:
        raise ValueError(f"{spec!r} is not a Hub 'org/name[/subdir]' spec.")
    return f"{parts[0]}/{parts[1]}", "/".join(parts[2:])


def is_local(pre_dir: str) -> bool:
    return Path(pre_dir).expanduser().exists()


def resolve_pre_dir(pre_dir: str) -> str:
    p = Path(pre_dir).expanduser()
    if not p.is_dir():
        raise FileNotFoundError(
            f"pre_dir {pre_dir!r} does not exist. Preprocessed data is not "
            f"downloaded on demand; fetch it first, e.g.\n"
            f"  hf download stanford-star/the-join-preprocessed --repo-type dataset "
            f"--local-dir {pre_dir}\n"
            f"(see examples/README.md for fetching only the dbs you need)"
        )
    return str(p)


def _is_complete(dataset_dir: Path) -> bool:
    meta_path = dataset_dir / "meta.json"
    if not meta_path.exists():
        return False
    try:
        embs = json.loads(meta_path.read_text()).get("text_embeddings", {})
    except Exception:  # noqa: BLE001
        return False
    return bool(embs) and all((dataset_dir / e["file"]).exists() for e in embs.values())


def list_datasets(pre_dir: str) -> list[str]:
    p = Path(resolve_pre_dir(pre_dir))
    return sorted(d.name for d in p.iterdir() if d.is_dir() and _is_complete(d))


def read_meta(pre_dir: str, db: str) -> dict:
    return json.loads((Path(pre_dir).expanduser() / db / "meta.json").read_text())


import multiprocessing as _mp
import resource as _resource

import torch

try:
    _mp.set_start_method("fork")
except RuntimeError:
    pass
torch.multiprocessing.set_sharing_strategy("file_descriptor")
_soft, _hard = _resource.getrlimit(_resource.RLIMIT_NOFILE)
if _soft < _hard:
    _resource.setrlimit(_resource.RLIMIT_NOFILE, (_hard, _hard))


@cache
def _load_column_index(db_name: str, pre_dir: str) -> dict:
    pre_dir = Path(pre_dir).expanduser()
    column_index_path = f"{pre_dir}/{db_name}/column_index.json"
    with open(column_index_path) as f:
        return json.load(f)


def get_column_index(
    column_name: str, table_name: str, db_name: str, pre_dir: str
) -> int:
    column_index = _load_column_index(db_name, pre_dir)
    target = f"{column_name} of {table_name}"

    if target not in column_index:
        raise ValueError(
            f'no column "{target}" in {db_name} '
            "(dropped in preprocessing, most likely constant)"
        )

    return column_index[target]
