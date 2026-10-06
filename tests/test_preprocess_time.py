import polars as pl
import pytest
import yaml
from rt.rustler import preprocess

WARNING = "warning: time column"


def with_events_timestamp(ds, dtype):
    path = ds / "db" / "events.parquet"
    events = pl.read_parquet(path)
    events.with_columns(pl.col("timestamp").cast(dtype)).write_parquet(path)
    return ds


@pytest.mark.parametrize(
    "dtype", [pl.Datetime("us"), pl.Date], ids=["datetime", "date"]
)
def test_a_date_or_datetime_time_column_is_accepted_quietly(
    synthetic_dataset, tmp_path, capfd, dtype
):
    ds = with_events_timestamp(synthetic_dataset, dtype)
    preprocess(str(ds), str(tmp_path / "out"), skip_tasks=True)
    assert WARNING not in capfd.readouterr().err


def test_a_string_time_column_is_flagged(synthetic_dataset, tmp_path, capfd):
    ds = with_events_timestamp(synthetic_dataset, pl.String)
    preprocess(str(ds), str(tmp_path / "out"), skip_tasks=True)
    assert (
        "warning: time column timestamp of events (Db) has dtype str"
        in capfd.readouterr().err
    )


def test_a_missing_time_column_is_flagged(synthetic_dataset, tmp_path, capfd):
    manifest_path = synthetic_dataset / "manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["tables"]["events"]["time_col"] = "occurred_at"
    manifest_path.write_text(yaml.safe_dump(manifest))
    preprocess(str(synthetic_dataset), str(tmp_path / "out"), skip_tasks=True)
    assert (
        "warning: time column occurred_at is not a column of events (Db)"
        in capfd.readouterr().err
    )
