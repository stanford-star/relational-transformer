import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file

from rt.data import resolve_repo

try:
    _RT_VERSION = version("relational-transformer")
except PackageNotFoundError:
    _RT_VERSION = None

_HF_UA = {"library_name": "relational-transformer", "library_version": _RT_VERSION}

CONFIG_FILE = "config.json"
MODEL_FILE = "model.safetensors"
MODEL_DIM_KEYS = ("num_blocks", "d_model", "d_text", "num_heads", "d_ff")


def save_model(state_dict, path, metadata: dict | None = None) -> None:
    meta = {str(k): str(v) for k, v in (metadata or {}).items()}
    save_file(state_dict, str(path), metadata=meta or None)


def load_model(path):
    sd = load_file(str(path))
    for k in [k for k in sd if ".boolean" in k]:
        del sd[k]
    return sd


def _compat(config: dict) -> None:
    if "embedder" not in config and "embedding_model" in config:
        config["embedder"] = config.pop("embedding_model")


def resolve_checkpoint(spec, *, revision: str | None = None) -> tuple[dict, Path]:
    p = Path(spec).expanduser()
    if p.is_file():
        cfg_path = p.with_name(CONFIG_FILE)
        config = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        _compat(config)
        return config, p
    if p.is_dir():
        d = p
    else:
        repo_id, subdir = resolve_repo(spec)
        local = snapshot_download(
            repo_id=repo_id,
            revision=revision,
            allow_patterns=[f"{subdir}/*"] if subdir else None,
            **_HF_UA,
        )
        d = Path(local) / subdir if subdir else Path(local)
    config = json.loads((d / CONFIG_FILE).read_text())
    _compat(config)
    return config, d / config.get("checkpoint_file", MODEL_FILE)
