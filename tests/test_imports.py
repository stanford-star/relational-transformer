import subprocess
import sys


def test_preprocess_import_is_light():
    code = (
        "import sys, rt.preprocess; "
        "print([m for m in ('transformers', 'sentence_transformers', 'rt.model') "
        "if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip() == "[]"
