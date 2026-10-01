import subprocess
import sys
import textwrap


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        check=False,
    )


def test_rt_preprocess_imports():
    r = _run(
        """
        import sys, types
        for name, attrs in [
            ("orjson", {"loads": lambda *a, **k: None}),
            ("ml_dtypes", {"bfloat16": object()}),
            ("sentence_transformers", {"SentenceTransformer": object()}),
        ]:
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m
        import rt.preprocess
        assert hasattr(rt.preprocess, "TextEmbedder")
        print("ok")
        """
    )
    assert r.returncode == 0, r.stderr
