import json
import shutil
from pathlib import Path

import polars as pl
import pytest
from rt.rustler import preprocess

BYTES = ("nodes.rkyv", "offsets.rkyv", "p2f_adj.rkyv", "text.json")
# written from HashMaps, so their key order is not stable across runs
JSON = ("table_info.json", "column_index.json")
USER_ID_FILES = (
    "db/users.parquet",
    "db/events.parquet",
    "tasks/spend/train.parquet",
    "tasks/spend/val.parquet",
    "tasks/spend/test.parquet",
)
N_USERS = 10


def rekeyed(src: Path, dst: Path, key, reverse_users: bool = False) -> Path:
    """Copy of `src` with every user id (users' pkey, events' fkey, the task's
    entity column) mapped through `key`; optionally users stored back to front."""
    shutil.copytree(src, dst)
    for rel in USER_ID_FILES:
        df = pl.read_parquet(dst / rel)
        ids = [None if k is None else key(k) for k in df["user_id"].to_list()]
        df = df.with_columns(pl.Series("user_id", ids))
        if reverse_users and rel == "db/users.parquet":
            df = df.reverse()
        df.write_parquet(dst / rel)
    return dst


def outputs(dataset: Path, out: Path) -> dict:
    preprocess(str(dataset), str(out))
    (produced,) = [d for d in out.iterdir() if d.is_dir()]
    res = {n: (produced / n).read_bytes() for n in BYTES}
    res |= {n: json.loads((produced / n).read_text()) for n in JSON}
    return res


def test_preprocessing_is_deterministic(synthetic_dataset_with_external_task, tmp_path):
    ds = synthetic_dataset_with_external_task
    assert outputs(ds, tmp_path / "a") == outputs(ds, tmp_path / "b")


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(lambda k: k + 1, id="one-based"),
        pytest.param(lambda k: 10 * k + 7, id="sparse"),
        pytest.param(lambda k: k + 10_000_000, id="beyond-the-node-count"),
        pytest.param(lambda k: f"user-{k:03d}", id="strings"),
    ],
)
def test_foreign_keys_resolve_through_the_primary_key(
    synthetic_dataset_with_external_task, tmp_path, key
):
    ds = synthetic_dataset_with_external_task
    expected = outputs(ds, tmp_path / "base_out")
    actual = outputs(rekeyed(ds, tmp_path / "rekeyed", key), tmp_path / "out")
    assert actual == expected


def test_parent_rows_need_not_be_in_key_order(
    synthetic_dataset_with_external_task, tmp_path
):
    ds = synthetic_dataset_with_external_task
    # the same reversed users table, keyed by row position: what the keyed copy
    # below must resolve to
    positional = rekeyed(
        ds, tmp_path / "positional", lambda k: N_USERS - 1 - k, reverse_users=True
    )
    keyed = rekeyed(ds, tmp_path / "keyed", lambda k: k, reverse_users=True)
    assert outputs(keyed, tmp_path / "keyed_out") == outputs(
        positional, tmp_path / "positional_out"
    )


def test_unmatched_foreign_keys_are_dropped_with_a_warning(
    synthetic_dataset_with_external_task, tmp_path, capfd
):
    ds = synthetic_dataset_with_external_task

    def with_event_user(dst, value):
        dst = rekeyed(ds, dst, lambda k: k)
        events = pl.read_parquet(dst / "db/events.parquet")
        ids = events["user_id"].to_list()
        ids[3] = value
        events.with_columns(pl.Series("user_id", ids)).write_parquet(
            dst / "db/events.parquet"
        )
        return dst

    expected = outputs(with_event_user(tmp_path / "null", None), tmp_path / "a")
    capfd.readouterr()
    actual = outputs(with_event_user(tmp_path / "dangling", 999), tmp_path / "b")
    assert "1 value(s) of user_id in events (Db) reference no row of users" in (
        capfd.readouterr().err
    )
    assert actual == expected


def test_a_duplicate_primary_key_is_an_error(
    synthetic_dataset_with_external_task, tmp_path
):
    ds = rekeyed(
        synthetic_dataset_with_external_task,
        tmp_path / "dup",
        lambda k: 100 + min(k, 8),
    )
    with pytest.raises(BaseException, match="duplicate value 108"):
        preprocess(str(ds), str(tmp_path / "out"))
