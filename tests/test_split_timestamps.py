import json
import subprocess
import sys
import textwrap

import pytest

_PROBE = """
import json, sys, types
from pathlib import Path

import pandas as pd

pre_dir = Path(sys.argv[1])
(pre_dir / "demo-db").mkdir(parents=True, exist_ok=True)
(pre_dir / "demo-db" / "meta.json").write_text(json.dumps({"source": "demo-src"}))


class _Dataset:
    val_timestamp = pd.Timestamp(sys.argv[2])
    test_timestamp = pd.Timestamp(sys.argv[3])


relbench = types.ModuleType("relbench")
relbench.load_dataset = lambda source: _Dataset()
sys.modules["relbench"] = relbench

from rt.data.datasets import _split_timestamps

print(json.dumps(_split_timestamps("demo-db", str(pre_dir))))
"""


def _split_timestamps_under(tz: str, tmp_path, val: str, test: str) -> dict[str, int]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(_PROBE), str(tmp_path), val, test],
        capture_output=True,
        text=True,
        env={"TZ": tz, "PATH": "/usr/bin:/bin"},
        cwd=str(tmp_path),
        check=False,
    )
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("tz", ["UTC", "America/Los_Angeles", "Asia/Kolkata"])
def test_split_timestamps_are_utc_under_any_tz(tz, tmp_path):
    got = _split_timestamps_under(
        tz, tmp_path / tz.replace("/", "-"), "2020-01-01", "2021-01-01"
    )
    assert got == {"val": 1577836800, "test": 1609459200}


def test_split_timestamps_reject_tz_aware(tmp_path):
    r = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(_PROBE),
            str(tmp_path),
            "2020-01-01T00:00:00+05:30",
            "2021-01-01",
        ],
        capture_output=True,
        text=True,
        env={"TZ": "UTC", "PATH": "/usr/bin:/bin"},
        cwd=str(tmp_path),
        check=False,
    )
    assert r.returncode != 0
    assert "tz-aware split timestamp" in r.stderr
