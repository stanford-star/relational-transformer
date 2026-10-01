import os

import pytest

from rt.data import stage_paths


@pytest.fixture
def other_device(monkeypatch):
    real = os.stat

    def stat(path, *a, **k):
        st = real(path, *a, **k)
        if "/localdisk" in str(path):
            st = os.stat_result(tuple(st)[:2] + (st.st_dev + 1,) + tuple(st)[3:])
        return st

    monkeypatch.setattr("rt.data.stage.os.stat", stat)


def make_pre(root, name, dbs=3):
    pre = root / name
    for i in range(dbs):
        (pre / f"db{i}").mkdir(parents=True)
        (pre / f"db{i}" / "nodes.rkyv").write_bytes(bytes([i]) * 1000)
    (pre / "index.json").write_text("{}")
    return pre


def test_staged_paths_are_relocated_copies(tmp_path, other_device):
    pre = make_pre(tmp_path / "src", "the-join")
    ev = make_pre(tmp_path / "src", "relbench", dbs=1)
    calls = []
    out = stage_paths(
        str(tmp_path / "localdisk"),
        [str(pre), str(ev)],
        local_rank=0,
        barrier=lambda: calls.append(1),
    )
    assert out == [
        str(tmp_path / "localdisk" / "the-join"),
        str(tmp_path / "localdisk" / "relbench"),
    ]
    assert calls == [1]
    for name, n in (("the-join", 3), ("relbench", 1)):
        d = tmp_path / "localdisk" / name
        assert (d / ".staged").exists()
        assert sorted(p.name for p in d.iterdir() if p.name.startswith("db")) == [
            f"db{i}" for i in range(n)
        ]
        assert (d / "db0" / "nodes.rkyv").read_bytes() == bytes([0]) * 1000


def test_a_marker_skips_the_copy_and_other_ranks_only_wait(tmp_path, other_device):
    pre = make_pre(tmp_path / "src", "the-join")
    stage_paths(
        str(tmp_path / "localdisk"), [str(pre)], local_rank=0, barrier=lambda: None
    )
    marker = tmp_path / "localdisk" / "the-join" / ".staged"
    before = marker.stat().st_mtime_ns
    (pre / "db0" / "nodes.rkyv").write_bytes(b"changed")
    stage_paths(
        str(tmp_path / "localdisk"), [str(pre)], local_rank=0, barrier=lambda: None
    )
    assert marker.stat().st_mtime_ns == before
    assert (
        tmp_path / "localdisk" / "the-join" / "db0" / "nodes.rkyv"
    ).read_bytes() != b"changed"
    out = stage_paths(
        str(tmp_path / "localdisk2"), [str(pre)], local_rank=1, barrier=lambda: None
    )
    assert out == [str(tmp_path / "localdisk2" / "the-join")]
    assert not (tmp_path / "localdisk2").exists()


def test_env_vars_expand_and_same_filesystem_is_refused(tmp_path, monkeypatch):
    pre = make_pre(tmp_path / "src", "the-join")
    monkeypatch.setenv("RT_TEST_STAGE", str(tmp_path / "notlocal"))
    with pytest.raises(AssertionError, match="same filesystem"):
        stage_paths("$RT_TEST_STAGE", [str(pre)], local_rank=0, barrier=lambda: None)


def test_a_missing_source_is_refused(tmp_path, other_device):
    with pytest.raises(AssertionError, match="nothing to stage"):
        stage_paths(
            str(tmp_path / "localdisk"),
            [str(tmp_path / "absent")],
            local_rank=0,
            barrier=lambda: None,
        )
