import json
from datetime import datetime, timedelta

import polars as pl
import pytest
import yaml

from rt import RelationalTransformer
from rt.model import CONFIG_FILE, MODEL_FILE, save_model

TINY_DIMS = {"num_blocks": 2, "d_model": 16, "d_text": 8, "num_heads": 2, "d_ff": 32}


@pytest.fixture(scope="session")
def tiny_dims() -> dict:
    return dict(TINY_DIMS)


@pytest.fixture()
def tiny_checkpoint(tmp_path, tiny_dims):
    src = RelationalTransformer(**tiny_dims, compile=False, materialize_attn_masks=True)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    save_model(src.state_dict(), ckpt / MODEL_FILE)
    (ckpt / CONFIG_FILE).write_text(
        json.dumps({"model": tiny_dims, "embedder": "test-embed"})
    )
    return ckpt, src


@pytest.fixture()
def synthetic_dataset(tmp_path):
    n_users, n_events = 10, 18
    users = pl.DataFrame(
        {
            "user_id": range(n_users),
            "name": [f"user {i}" for i in range(n_users)],
            "plan": ["free" if i % 3 else "pro" for i in range(n_users)],
            "credit": [round(1.5 * i, 2) for i in range(n_users)],
            "active": [i % 4 != 0 for i in range(n_users)],
        }
    )
    events = pl.DataFrame(
        {
            "event_id": range(n_events),
            "user_id": [i % n_users for i in range(n_events)],
            "kind": ["click" if i % 2 else "view" for i in range(n_events)],
            "amount": [float(i % 7) for i in range(n_events)],
            "timestamp": [
                datetime(2024, 1, 1) + timedelta(days=i)  # noqa: DTZ001
                for i in range(n_events)
            ],
        }
    )

    ds = tmp_path / "synth"
    (ds / "db").mkdir(parents=True)
    users.write_parquet(ds / "db" / "users.parquet")
    events.write_parquet(ds / "db" / "events.parquet")

    manifest = {
        "name": "synth",
        "tables": {
            "users": {"pkey": "user_id"},
            "events": {
                "pkey": "event_id",
                "time_col": "timestamp",
                "fkeys": {"user_id": "users"},
            },
        },
    }
    (ds / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return ds


@pytest.fixture()
def synthetic_dataset_with_external_task(synthetic_dataset):
    ds = synthetic_dataset
    tdir = ds / "tasks" / "spend"
    tdir.mkdir(parents=True)
    rows = {
        "train": [(i, datetime(2024, 1, 5), float(i)) for i in range(6)],  # noqa: DTZ001
        "val": [(i, datetime(2024, 1, 12), float(i)) for i in range(6, 8)],  # noqa: DTZ001
        "test": [(i, datetime(2024, 1, 16), float(i)) for i in range(8, 10)],  # noqa: DTZ001
    }
    for split, recs in rows.items():
        pl.DataFrame(
            {
                "user_id": [r[0] for r in recs],
                "timestamp": [r[1] for r in recs],
                "spend": [r[2] for r in recs],
            }
        ).write_parquet(tdir / f"{split}.parquet")
    (tdir / "manifest.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "spend",
                "kind": "external",
                "task_type": "regression",
                "entity_table": "users",
                "entity_col": "user_id",
                "target_col": "spend",
                "time_col": "timestamp",
                "remove_columns": [["events", "amount"]],
                "manifest_version": 1,
            }
        )
    )
    return ds
