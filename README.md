# Relational Transformer (RT)

This repo is the official implementation
of the **Relational Transformer (RT)** architecture
and the recipes to pretrain, fine-tune, inference and test-time scale
**Relational Foundation Models (RFMs)** based on RT.

The following results are fully reproducible from here:  
🥇 **RT-J** is the #1 in-context model on the [RelBench leaderboard](https://star-project.stanford.edu/relbench/leaderboard) at the time of submission.  
🥇 **RT-PluRel** is #1 on the combined system + model leaderboard on [RelArena-alpha](https://star-project.stanford.edu/relarena-alpha) at the time of submission.

This repo is linked to the following papers in the
[Stanford Tabular and Relational (STAR) project](https://star-project.stanford.edu) ecosystem:

| Paper | Venue | Implementation |
|---|---|---|
| [RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions](https://star-project.stanford.edu/rt-j) | NeurIPS 2026 | RT-J recipes + paper experiments: this repo. |
| [PluRel: Synthetic Data unlocks Scaling Laws for Relational Foundation Models](https://arxiv.org/abs/2602.04029) | ICML 2026 | Synthetic data generation + paper experiments: [`stanford-star/plurel`](https://github.com/stanford-star/plurel). RT-PluRel recipes: this repo.|
| [Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](https://arxiv.org/abs/2510.06377) | ICLR 2026 | Legacy architecture + paper experiments: this repo @ [`rt-v1`](https://github.com/stanford-star/relational-transformer/tree/rt-v1) |

## Installation

Install the `rt` package (the model plus the native Rust data engine) from
GitHub. It builds the extension from source, so you need a
[Rust toolchain](https://rustup.rs) and Python 3.11+:

```bash
pip install "git+https://github.com/stanford-star/relational-transformer.git"
```

## Quickstart

The quickest way to try a released checkpoint is on a RelBench
database already preprocessed into RT's tensor format on the Hub. The example below predicts whether an F1 driver fails to finish a race (`driver-dnf`) with a released RT-J checkpoint:

```python
import os

# flex_attention's compiled kernel is CUDA-only; run it eager on CPU/MPS
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
from huggingface_hub import snapshot_download

from rt import RelationalTransformer
from rt.eval import build_evaluator
from rt.data import get_tasks

device = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# 1. download one RelBench database, already preprocessed into RT's tensor format
pre_dir = snapshot_download(
    "stanford-star/relbench-preprocessed",
    repo_type="dataset",
    allow_patterns="rel-f1/*",
)

# 2. load a pretrained checkpoint (RT-J here)
model = RelationalTransformer.from_pretrained(
    "stanford-star/rt-j", device=device
).to(torch.bfloat16)
cfg = model.config

# 3. build an evaluator for one task and predict zero-shot for 5 test rows.
#    the evaluator samples each row's context from the preprocessed DB;
tasks = get_tasks(pre_dir, [("rel-f1", "driver-dnf")], ("test",))
ev = build_evaluator(
    tasks, pre_dir,
    embedder=cfg["embedder"], d_text=cfg["d_text"], device=device,
    ctx_size_list=[128], local_ctx_size=64, bfs_width=32, prefer_latest=True,
    num_walks=10_000, walk_length=20, tokens_per_gpu=2**18,
    items_per_task=5, num_workers=0, prefetch_factor=2,
    shuffle_seed=0, context_seed=0, mmap_populate=True, vector_db_path=None,
    db_cutoff=None,
)

# evaluate_raw yields one (task, ctx, labels, preds, n) per task
results = ev.evaluate_raw([(model, "")], [128])
_task, _ctx, _labels, out, _n = next(iter(results))
preds = torch.sigmoid(torch.tensor(out[""], dtype=torch.float32))
print("driver-dnf probability:", [round(p, 3) for p in preds.tolist()])
```

> [!NOTE]
> `items_per_task=5` and a 128-cell context keep this demo small. On a CPU it
> still takes tens of minutes: `flex_attention` has no fused bfloat16 CPU kernel,
> so the forward pass dominates. On a GPU it is seconds, and you can raise
> `ctx_size_list` toward RT-J's training context of 8192 (with
> `local_ctx_size <= ctx_size`) for full accuracy over the whole test split.

## Released checkpoints

| Checkpoint | Paper | Loader |
|---|---|---|
| [`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j) | RT-J | `RelationalTransformer.from_pretrained` |
| [`stanford-star/rt-plurel`](https://huggingface.co/stanford-star/rt-plurel) | PluRel | [`examples/eval/legacy.py`](examples/eval/legacy.py) |
| [`stanford-star/rt-v1`](https://huggingface.co/stanford-star/rt-v1) | ICLR 2026 | [`examples/eval/legacy.py`](examples/eval/legacy.py) |

The context is embedded with the vectors on disk in `pre_dir`, so a checkpoint
evaluated under a different embedder scores garbage rather than failing;
`rt.eval.main` asserts the `embedder` and `d_text` match the checkpoint.
Architecture, recipe, protocol, licence and limitations are on each Hub card,
kept here under [`docs/cards/`](docs/cards/).

## Bring your own database

Point RT at your **own** database, define a
prediction task, and infer with a released checkpoint: the
[fully worked notebook](byod/colab.ipynb)
([open in Colab](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/byod/colab.ipynb))
runs the whole flow end-to-end on your database, or on the bundled demo.

For more than a first look, [`examples/`](examples/README.md) runs the full
recipes on your own data in RelBench format: in-context prediction with a
frozen checkpoint, or per-task fine-tuning.

## Development

We use [pixi](https://pixi.sh) to manage one self-contained
environment (Python, PyTorch + CUDA, Rust, and all dependencies), built on first use.

```bash
git clone https://github.com/stanford-star/relational-transformer.git
cd relational-transformer
pixi run pytest                              # the test suite
pixi run python -m examples.eval.plan        # or examples.pretrain.phases, ...
```

## Documentation

| Guide | Description |
|---|---|
| [Downloads](docs/downloads.md) | Bulk-download raw data, preprocessed data, and checkpoints from HuggingFace |
| [Preprocess](docs/preprocess.md) | Convert RelBench-format databases into RT's on-disk format |
| [Inference](docs/inference.md) | Run a trained checkpoint; evaluate, engineer, tune, and ensemble contexts |
| [Pretrain](docs/train.md) | Train RT from scratch, single-GPU to multi-node |
| [Context visualization](docs/ctx_viz.md) | Inspect the contexts sampled for each row |
| [Examples](examples/README.md) | Preprocess, pretrain, evaluate, in-context and fine-tune, on RelBench or your own data |
| [Cards](docs/cards/README.md) | Model and dataset cards |

## Citation

If you use this repo, please cite the following papers:

```bibtex
@inproceedings{ranjan2026rtj,
    title={{RT-J}: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions},
    author={Rishabh Ranjan and Vignesh Kothapalli and Harshvardhan Agarwal and Charilaos Kanatsoulis and Roshan Upendra and Tom Palczewski and Carlos Guestrin and Jure Leskovec},
    booktitle={The Fortieth Annual Conference on Neural Information Processing Systems},
    year={2026}
}

@inproceedings{ranjan2026relational,
    title={Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data},
    author={Rishabh Ranjan and Valter Hudovernik and Mark Znidar and Charilaos Kanatsoulis and Roshan Upendra and Mahmoud Mohammadi and Joe Meyer and Tom Palczewski and Carlos Guestrin and Jure Leskovec},
    booktitle={The Fourteenth International Conference on Learning Representations},
    year={2026}
}
```

Additionally, if you use **RT-PluRel** please also cite:

```bibtex
@inproceedings{kothapalli2026plurel,
    title={{PluRel}: Synthetic Data unlocks Scaling Laws for Relational Foundation Models},
    author={Vignesh Kothapalli and Rishabh Ranjan and Valter Hudovernik and Vijay Prakash Dwivedi and Johannes Hoffart and Carlos Guestrin and Jure Leskovec},
    booktitle={Forty-third International Conference on Machine Learning},
    year={2026}
}
```

## License

MIT, see [LICENSE](LICENSE). The released checkpoints and datasets carry their
own licences; "the Join" is assembled from third-party databases that keep
theirs.
