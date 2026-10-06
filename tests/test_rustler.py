import json
import threading
import time

import ml_dtypes
import numpy as np
import rt.rustler as r
from rt.rustler import Sampler, preprocess

from rt.data.resolve import get_column_index
from rt.data.tasks import get_tasks


def test_extension_symbols():
    assert hasattr(r, "Sampler")
    assert hasattr(r, "preprocess")


def test_preprocess_end_to_end(synthetic_dataset, tmp_path):
    out = tmp_path / "out"
    preprocess(str(synthetic_dataset), str(out), skip_tasks=True)

    (produced,) = [d for d in out.iterdir() if d.is_dir()]
    files = {p.name for p in produced.rglob("*") if p.is_file()}
    assert {"nodes.rkyv", "table_info.json", "column_index.json"} <= files


def test_remove_columns_reaches_the_sampler_for_a_non_autocomplete_task(
    synthetic_dataset_with_external_task, tmp_path
):
    out = tmp_path / "out"
    preprocess(str(synthetic_dataset_with_external_task), str(out))
    (produced,) = [d for d in out.iterdir() if d.is_dir()]
    pre_dir, db_name = str(out), produced.name

    meta = json.loads((produced / "meta.json").read_text())
    (task_meta,) = [t for t in meta["tasks"] if t["name"] == "spend"]
    assert task_meta["kind"] == "external"
    assert task_meta["remove_columns"] == [["events", "amount"]]

    (task,) = get_tasks(pre_dir, [[db_name, "spend"]], ["train"])
    assert task.leakage_columns == (("events", "amount"),)

    drop_idx = get_column_index("amount", "events", db_name, pre_dir)
    target_idx = get_column_index(task.target_column, task.table_name, db_name, pre_dir)
    assert drop_idx != target_idx
    kind_idx = get_column_index("kind", "events", db_name, pre_dir)
    assert drop_idx != kind_idx


def test_sampler_drops_remove_columns_on_the_targets_horizon(
    synthetic_dataset_with_external_task, tmp_path
):
    out = tmp_path / "out"
    preprocess(str(synthetic_dataset_with_external_task), str(out))
    (produced,) = [d for d in out.iterdir() if d.is_dir()]
    pre_dir, db_name, d_text, embedder = str(out), produced.name, 8, "test-embed"

    n_text = len(json.loads((produced / "text.json").read_text()))
    (produced / f"text_emb_{embedder}.bin").write_bytes(
        np.zeros((n_text, d_text), dtype=ml_dtypes.bfloat16).tobytes()
    )

    (task,) = get_tasks(pre_dir, [[db_name, "spend"]], ["train"])
    info = json.loads((produced / "table_info.json").read_text())
    span = info[f"{task.table_name}:Train"]
    drop_idx = get_column_index("amount", "events", db_name, pre_dir)
    sibling_idx = get_column_index("kind", "events", db_name, pre_dir)
    target_idx = get_column_index(task.target_column, task.table_name, db_name, pre_dir)

    def context_columns(columns_to_drop):
        sampler = Sampler(
            [(db_name, task.table_name, span["node_idx_offset"], span["num_nodes"])],
            0,
            0,
            1,
            [64],
            [8],
            0,
            10,
            [False],
            0.0,
            embedder,
            pre_dir,
            d_text,
            0,
            0,
            [target_idx],
            [columns_to_drop],
            [None],
            -1,
            True,
            False,
            0,
            True,
            False,
            10.0,
            None,
        )
        seq_len = 64
        cells = []
        for batch_idx in range(sampler.num_items):
            batch = dict(sampler.batch_py(batch_idx, 2, seq_len))
            cols = batch["col_name_idxs"].reshape(-1, seq_len)
            times = batch["timestamps"].reshape(-1, seq_len)
            pads = batch["is_padding"].reshape(-1, seq_len).astype(bool)
            for row_cols, row_times, row_pads in zip(cols, times, pads):
                target_ts = int(row_times[0])
                for col, ts, pad in zip(row_cols, row_times, row_pads):
                    if not pad:
                        cells.append((int(col), int(ts), target_ts))
        return cells

    drops = [
        get_column_index(col, table, db_name, pre_dir)
        for table, col in task.leakage_columns
    ]
    assert drops == [drop_idx]

    without = context_columns([])
    on_horizon = [c for c in without if c[0] == drop_idx and c[1] == c[2]]
    in_past = [c for c in without if c[0] == drop_idx and c[1] != c[2]]
    assert on_horizon, "fixture never puts the column on the target's horizon"
    assert in_past, "fixture never puts the column on a past row"

    with_drop = context_columns(drops)
    assert not [c for c in with_drop if c[0] == drop_idx and c[1] == c[2]]
    assert [c for c in with_drop if c[0] == drop_idx and c[1] != c[2]]
    seen = {c[0] for c in with_drop}
    assert sibling_idx in seen
    assert target_idx in seen


def test_prefer_latest_biases_the_fallback_tier_toward_recent_rows(
    synthetic_dataset_with_external_task, tmp_path
):
    out = tmp_path / "out"
    preprocess(str(synthetic_dataset_with_external_task), str(out))
    (produced,) = [d for d in out.iterdir() if d.is_dir()]
    pre_dir, db_name, d_text, embedder = str(out), produced.name, 8, "test-embed"
    n_text = len(json.loads((produced / "text.json").read_text()))
    (produced / f"text_emb_{embedder}.bin").write_bytes(
        np.zeros((n_text, d_text), dtype=ml_dtypes.bfloat16).tobytes()
    )
    (task,) = get_tasks(pre_dir, [[db_name, "spend"]], ["train"])
    info = json.loads((produced / "table_info.json").read_text())
    span = info[f"{task.table_name}:Train"]
    target_idx = get_column_index(task.target_column, task.table_name, db_name, pre_dir)

    def past_offsets(prefer_latest):
        sampler = Sampler(
            [(db_name, task.table_name, span["node_idx_offset"], span["num_nodes"])],
            0,
            0,
            1,
            [64],
            [8],
            0,
            10,
            [prefer_latest],
            0.0,
            embedder,
            pre_dir,
            d_text,
            0,
            0,
            [target_idx],
            [[]],
            [None],
            -1,
            True,
            False,
            0,
            True,
            False,
            10.0,
            None,
        )
        seq_len = 64
        gaps = []
        for batch_idx in range(sampler.num_items):
            batch = dict(sampler.batch_py(batch_idx, 2, seq_len))
            times = batch["timestamps"].reshape(-1, seq_len)
            pads = batch["is_padding"].reshape(-1, seq_len).astype(bool)
            for row_times, row_pads in zip(times, pads):
                target_ts = int(row_times[0])
                for ts, pad in zip(row_times, row_pads):
                    if not pad and int(ts) < target_ts:
                        gaps.append(target_ts - int(ts))
        return gaps

    late, uniform = past_offsets(True), past_offsets(False)
    assert late and uniform, "fixture never quotes a strictly-past row"
    assert sum(late) / len(late) <= sum(uniform) / len(uniform), (
        f"prefer_latest=True mean gap {sum(late) / len(late):.1f} is not closer to "
        f"the target than False's {sum(uniform) / len(uniform):.1f}"
    )


def test_batch_for_nodes_releases_the_gil(
    synthetic_dataset_with_external_task, tmp_path
):
    out = tmp_path / "out"
    preprocess(str(synthetic_dataset_with_external_task), str(out))
    (produced,) = [d for d in out.iterdir() if d.is_dir()]
    pre_dir, db_name, d_text, embedder = str(out), produced.name, 8, "test-embed"
    n_text = len(json.loads((produced / "text.json").read_text()))
    (produced / f"text_emb_{embedder}.bin").write_bytes(
        np.zeros((n_text, d_text), dtype=ml_dtypes.bfloat16).tobytes()
    )
    (task,) = get_tasks(pre_dir, [[db_name, "spend"]], ["train"])
    info = json.loads((produced / "table_info.json").read_text())
    span = info[f"{task.table_name}:Train"]
    target_idx = get_column_index(task.target_column, task.table_name, db_name, pre_dir)
    sampler = Sampler(
        [(db_name, task.table_name, span["node_idx_offset"], span["num_nodes"])],
        0,
        0,
        1,
        [64],
        [8],
        10_000,
        20,
        [False],
        0.0,
        embedder,
        pre_dir,
        d_text,
        0,
        0,
        [target_idx],
        [[]],
        [None],
        -1,
        True,
        False,
        0,
        True,
        False,
        10.0,
        None,
    )

    def rows(n):
        return [span["node_idx_offset"] + i % span["num_nodes"] for i in range(n)]

    tic = time.perf_counter()
    sampler.batch_for_nodes_py(rows(512), 0, 64)
    per_row = (time.perf_counter() - tic) / 512
    n = int(min(max(1.5 / per_row, 512), 20_000))

    call = {}

    def sample():
        call["enter"] = time.perf_counter()
        call["out"] = sampler.batch_for_nodes_py(rows(n), 0, 64)
        call["exit"] = time.perf_counter()

    thread = threading.Thread(target=sample)
    thread.start()
    wakes = []
    while thread.is_alive():
        time.sleep(0.01)
        wakes.append(time.perf_counter())
    thread.join()
    inside = sum(call["enter"] < w < call["exit"] for w in wakes)
    assert inside >= 5, (
        f"the main thread woke {inside} times inside a "
        f"{call['exit'] - call['enter']:.3f}s batch_for_nodes_py call: the sampler "
        f"holds the GIL while it samples"
    )
