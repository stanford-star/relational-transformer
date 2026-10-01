import json
from pathlib import Path


def load_table_info(pre_dir: str, db: str) -> dict:
    path = Path(pre_dir).expanduser() / db / "table_info.json"
    with open(path) as f:
        return json.load(f)


def get_table_splits(table_info: dict, table_name: str) -> dict[str, dict]:
    splits = {}
    for split in ["train", "val", "test", "db"]:
        key = f"{table_name}:{split.capitalize()}"
        if key in table_info:
            splits[split] = table_info[key]
    return splits


def table_offset_and_len(pre_dir: str, db: str, table_name: str) -> tuple[int, int]:
    splits_info = get_table_splits(load_table_info(pre_dir, db), table_name)
    assert splits_info, f"{db}/{table_name} not in table_info.json"
    sorted_offsets = sorted(
        (info["node_idx_offset"], info["num_nodes"]) for info in splits_info.values()
    )
    for (off, n), (nxt, _) in zip(sorted_offsets, sorted_offsets[1:]):
        assert off + n == nxt, (
            f"non-contiguous node_idxs across splits for {db}/{table_name}: "
            f"{sorted_offsets}"
        )
    min_offset = sorted_offsets[0][0]
    total = sum(n for _, n in sorted_offsets)
    return min_offset, total
