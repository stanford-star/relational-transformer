import json

import torch

from rt import RelationalTransformer
from rt.model import (
    CONFIG_FILE,
    MODEL_FILE,
    load_rt_model,
    save_model,
)


def test_from_pretrained_local(tiny_checkpoint):
    ckpt, src = tiny_checkpoint
    model = RelationalTransformer.from_pretrained(ckpt, device="cpu")
    assert isinstance(model, RelationalTransformer)
    assert model.config["embedder"] == "test-embed"
    s1, s2 = src.state_dict(), model.state_dict()
    assert s1.keys() == s2.keys()
    assert all(torch.equal(s1[k], s2[k]) for k in s1)


def test_load_rt_model_backcompat(tiny_checkpoint):
    ckpt, _ = tiny_checkpoint
    model, config = load_rt_model(str(ckpt))
    assert isinstance(model, RelationalTransformer)
    assert config["embedder"] == "test-embed"


def test_from_pretrained_model_kwargs(tmp_path, tiny_dims):
    src = RelationalTransformer(**tiny_dims, compile=False, materialize_attn_masks=True)
    save_model(src.state_dict(), tmp_path / MODEL_FILE)
    (tmp_path / CONFIG_FILE).write_text(json.dumps({"embedder": "x"}))
    model = RelationalTransformer.from_pretrained(tmp_path, **tiny_dims)
    assert model.config["embedder"] == "x"


def test_compile_true_builds(tiny_dims):
    m = RelationalTransformer(**tiny_dims, compile=True, materialize_attn_masks=True)
    assert callable(m.forward)
