import json

import yaml

from rt.data import make_db_task_list, make_plurel_db_task_list, write_db_task_list
from rt.data.tasks import resolve_db_task_list


def _write_db(pre_dir, name, tasks, column_index):
    d = pre_dir / name
    d.mkdir(parents=True)
    (d / "meta.json").write_text(json.dumps({"name": name, "tasks": tasks}))
    (d / "column_index.json").write_text(json.dumps(column_index))


def _task(name, entity_table, target_col, task_type, kind="autocomplete"):
    return {
        "name": name,
        "entity_table": entity_table,
        "target_col": target_col,
        "task_type": task_type,
        "kind": kind,
        "splits": [] if kind == "autocomplete" else ["train", "val", "test"],
    }


def test_make_db_task_list_kinds_and_dropped_targets(tmp_path):
    pre = tmp_path / "pre"
    _write_db(
        pre,
        "db-b",
        [
            _task("users-age", "users", "age", "regression"),
            _task("users-churn", "users", "churn", "binary_classification", "forecast"),
            _task("users-gone", "users", "gone", "regression"),
            _task("users-rec", "users", "items", "recommendation", "forecast"),
        ],
        {"age of users": 0, "churn of users-churn": 1},
    )
    _write_db(pre, "db-a", [_task("t-x", "t", "x", "regression")], {"x of t": 0})
    assert make_db_task_list(pre) == [
        ("db-a", "t-x"),
        ("db-b", "users-age"),
        ("db-b", "users-churn"),
    ]
    assert make_db_task_list(pre, kinds=("forecast",)) == [("db-b", "users-churn")]
    assert make_db_task_list(pre, kinds=("autocomplete",)) == [
        ("db-a", "t-x"),
        ("db-b", "users-age"),
    ]


def test_make_plurel_db_task_list(tmp_path):
    pre, raw = tmp_path / "pre", tmp_path / "raw"
    specs = {
        "plurel-3000": 2,
        "plurel-3001": 6,
        "plurel-3002": 0,
        "plurel-3003": 1,
    }
    for name, n_fkeys in specs.items():
        _write_db(
            pre,
            name,
            [
                _task("t-feature_0", "t", "feature_0", "regression"),
                _task("t-feature_1", "t", "feature_1", "binary_classification"),
                _task("t-feature_2", "t", "feature_2", "regression"),
                _task("t-feature_3", "t", "feature_3", "binary_classification"),
                _task("t-feature_4", "t", "feature_4", "regression"),
            ],
            {},
        )
        (raw / name).mkdir(parents=True)
        fkeys = {f"foreign_row_{i}": "u" for i in range(n_fkeys)}
        (raw / name / "manifest.yaml").write_text(
            yaml.safe_dump({"tables": {"t": {"fkeys": fkeys}, "u": {"fkeys": {}}}})
        )
        (raw / name / "scores.json").write_text(
            json.dumps(
                {
                    "t": {
                        "feature_0": {
                            "n_unique": 10,
                            "std": 1.0,
                            "is_source_node": False,
                        },
                        "feature_1": {
                            "n_unique": 2,
                            "n_classes": 2,
                            "majority_frac": 0.5,
                        },
                        "feature_2": {
                            "n_unique": 10,
                            "std": 1.0,
                            "is_source_node": True,
                        },
                        "feature_3": {
                            "n_unique": 2,
                            "n_classes": 2,
                            "majority_frac": 0.995,
                        },
                    }
                }
            )
        )
    pairs = make_plurel_db_task_list(pre, raw, num_dbs=2)
    assert pairs == [
        ("plurel-3000", "t-feature_0"),
        ("plurel-3000", "t-feature_1"),
        ("plurel-3000", "t-feature_4"),
        ("plurel-3002", "t-feature_0"),
        ("plurel-3002", "t-feature_1"),
        ("plurel-3002", "t-feature_4"),
    ]
    out = write_db_task_list(pairs, tmp_path / "out" / "list.json")
    assert resolve_db_task_list(out) == pairs
