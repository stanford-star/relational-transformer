import json

import pytest

from rt.eval import check_embedder
from rt.model import CONFIG_FILE, resolve_checkpoint

CONFIG = {"embedder": "all-MiniLM-L12-v2", "d_text": 384}


def test_matching_embedder_passes():
    check_embedder(CONFIG, "all-MiniLM-L12-v2", 384)


def test_mismatched_embedder_fails():
    with pytest.raises(AssertionError, match="embedder mismatch"):
        check_embedder(CONFIG, "all-mpnet-base-v2", 384)


def test_mismatched_d_text_fails():
    with pytest.raises(AssertionError, match="d_text mismatch"):
        check_embedder(CONFIG, "all-MiniLM-L12-v2", 768)


def test_missing_embedder_fails():
    with pytest.raises(AssertionError, match="no embedder"):
        check_embedder({"d_text": 384}, "all-MiniLM-L12-v2", 384)


def test_checkpoint_config_is_what_gets_checked(tiny_checkpoint, tiny_dims):
    ckpt, _ = tiny_checkpoint
    cfg_path = ckpt / CONFIG_FILE
    cfg_path.write_text(
        json.dumps({**json.loads(cfg_path.read_text()), "d_text": tiny_dims["d_text"]})
    )
    config, _ = resolve_checkpoint(str(ckpt))
    check_embedder(config, "test-embed", tiny_dims["d_text"])
    with pytest.raises(AssertionError, match="embedder mismatch"):
        check_embedder(config, "all-MiniLM-L12-v2", tiny_dims["d_text"])


def test_legacy_embedding_model_key_is_normalized(tmp_path):
    (tmp_path / CONFIG_FILE).write_text(
        json.dumps({"embedding_model": "all-MiniLM-L12-v2", "d_text": 384})
    )
    (tmp_path / "model.safetensors").touch()
    config, _ = resolve_checkpoint(str(tmp_path))
    check_embedder(config, "all-MiniLM-L12-v2", 384)
