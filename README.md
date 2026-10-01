# Relational Transformer (RT)

The official implementation of the **Relational Transformer (RT)**, an
architecture for **Relational Foundation Models (RFMs)** that predict directly
over relational databases (tables linked by foreign keys) and generalize
zero-shot to new databases, tasks, and schemas.

**This repository is RT-J**, the current model and the one to start from. The
two earlier papers are kept reproducible but are superseded; `main` tracks RT-J
only.

| Paper | Venue | Implementation | |
|---|---|---|---|
| RT-J: Large-Scale Pretraining of Relational Transformers for Context-Efficient Predictions | In progress | [`main`](https://github.com/stanford-star/relational-transformer) (this repo) | **start here** |
| [PluRel: Synthetic Data unlocks Scaling Laws for Relational Foundation Models](https://arxiv.org/abs/2602.04029) | ICML 2026 | [`stanford-star/plurel`](https://github.com/stanford-star/plurel) | superseded |
| [Relational Transformer: Toward Zero-Shot Foundation Models for Relational Data](https://arxiv.org/abs/2510.06377) | ICLR 2026 | [`rt-v1`](https://github.com/stanford-star/relational-transformer/tree/rt-v1) | superseded |

## Installation

Install the `rt` package (the model plus the native Rust data engine) from
GitHub. It builds the extension from source, so you need a
[Rust toolchain](https://rustup.rs) and Python 3.12+:

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

Three checkpoints are public on the
[`stanford-star`](https://huggingface.co/stanford-star) HuggingFace org. All
three were trained against the `all-MiniLM-L12-v2` text embedder with
`d_text=384`, which is the embedding published alongside the preprocessed data.

| Checkpoint | Paper | Loader | Notes |
|---|---|---|---|
| [`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j) | RT-J | `RelationalTransformer.from_pretrained` | **the current model**; 12 blocks, `d_model=512`, 8 heads, trained at context 8192 |
| [`stanford-star/rt-plurel`](https://huggingface.co/stanford-star/rt-plurel) | PluRel | [`examples/eval_legacy.py`](examples/eval_legacy.py) (`eval_plurel`) | the published architecture, kept verbatim in `rt.model.legacy` |
| [`stanford-star/rt-v1`](https://huggingface.co/stanford-star/rt-v1) | ICLR 2026 | [`examples/eval_legacy.py`](examples/eval_legacy.py) (`eval_v1`) | the published architecture, kept verbatim in `rt.model.legacy` |

The embedder is not a free choice at inference time: the context is embedded
with the vectors on disk in `pre_dir`, so evaluating a checkpoint under a
different embedder scores garbage rather than failing. `rt.eval.main` asserts
that the `embedder` and `d_text` you pass match the checkpoint's `config.json`.

The two legacy checkpoints also need differently preprocessed data (the
`legacy/` subdirectory of `stanford-star/relbench-preprocessed`) — see
[Inference](docs/inference.md). Fetch any of them with
[Downloads](docs/downloads.md); `from_pretrained` resolves `stanford-star/rt-j`
from the Hub on demand, so the quickstart above needs no manual download.

## Bring your own database

Point RT at your **own** database, define a
prediction task, and infer with a released checkpoint:

- **Colab, no setup**: the [fully worked notebook](byod/colab.ipynb)
  ([open in Colab](https://colab.research.google.com/github/stanford-star/relational-transformer/blob/main/byod/colab.ipynb))
  runs the whole flow end-to-end on your database (or the bundled demo).

## Reproducing the paper

The released checkpoints and this library are enough to reproduce RT-J's
*inference* numbers: every evaluation in the paper is `rt.eval.main` over a
preprocessed RelBench directory, and both the checkpoints and the data are
public. What the paper adds on top is a protocol — which context configuration
each task won, how many ensemble members, which baselines were featurized how
— and that protocol is [`reproduce/`](reproduce/), a cleaned subset of the
research directory the paper was produced from. It is research code with no
stability promise, published so the numbers are checkable.

**The reduced data behind every figure and table is committed**, in
[`reproduce/series/`](reproduce/series/): one JSON per arm, with the aggregate
metric at every context or ensemble size and the per-task value underneath it.
So are the two most expensive stages' results —
[`reproduce/tune/tuned_configs.json`](reproduce/tune/tuned_configs.json) (the
120-point per-task context grid, roughly 200 GPU-hours) and
[`reproduce/valtest/results.json`](reproduce/valtest/results.json) — and every
later stage reads them from there, so the grid never has to be re-run.

| What | Where | Needs |
|---|---|---|
| Check a published number against the released checkpoint | [`examples/eval.py`](examples/eval.py) at the context `tuned_configs.json` gives for the task | one GPU, minutes to hours |
| Re-derive a figure's series from finished runs | each stage's `reduce.py` / `collect.py` | no GPU, seconds |
| Re-run one evaluation on CPU | not practical — `flex_attention` has no fused bfloat16 CPU kernel | a GPU |
| Re-run the evaluations behind a figure | each stage's `plan.py` | hundreds of GPU-hours |
| Re-run the context grid | [`reproduce/tune/`](reproduce/tune/) | ~200 GPU-hours |
| Re-run the pretraining | [`docs/train.md`](docs/train.md) | thousands of GPU-hours |

The paper's own plotting scripts are **not** in this repository. They read our
private Weights & Biases projects with no cache in between, so for anyone
outside they would fail at the first API call —
[`reproduce/series/`](reproduce/series/) is the same data those plots were drawn
from, committed instead. Two things stay genuinely out of reach: the
**pretraining-ablation figures**, which compare five multi-day pretraining runs
whose checkpoints are not released (the knobs and the recipe are written down,
the runs are repeatable in principle), and six **appendix schematics** that
never had a generating script.

`reproduce/` carries no scheduler. The paper's runs went to a Slurm cluster, and
the partitions, QoS names and measured wall-clock limits that made that work
described one cluster on particular nights; they are gone. A stage instead
enumerates its jobs and runs them sequentially in-process, and
[`reproduce/launch.py`](reproduce/launch.py) is the one function to implement to
put them on your own scheduler. [`reproduce/README.md`](reproduce/README.md) is
the full map of what produces which figure, what each stage costs, what was
dropped from the research directory and why.

## Development

We use [pixi](https://pixi.sh) to manage one self-contained
environment (Python, PyTorch + CUDA, Rust, and all dependencies), built on first
use. There is nothing to build past `pixi install`: the native Rust extension is
compiled as part of the project's own editable install.

```bash
git clone https://github.com/stanford-star/relational-transformer.git
cd relational-transformer
pixi run pytest                        # the test suite
pixi run python examples/train.py      # or eval.py, preprocess.py, ...
```

## Documentation

| Guide | Description |
|---|---|
| [Downloads](docs/downloads.md) | Bulk-download raw data, preprocessed data, and checkpoints from HuggingFace |
| [Preprocess](docs/preprocess.md) | Convert RelBench-format databases into RT's on-disk format |
| [Inference](docs/inference.md) | Run a trained checkpoint; evaluate, engineer, tune, and ensemble contexts |
| [Pretrain](docs/train.md) | Train RT from scratch, single-GPU to multi-node |
| [Baselines](docs/baselines.md) | Run the `rel2tab` tabular baselines through the same eval path |
| [Context visualization](docs/context-visualization.md) | Inspect the context a config samples, in a browser |

There is no CLI: RT is a library, and a run is a script that calls it. Copy
something from [`examples/`](examples/) and edit it — every entry point takes
its arguments explicitly, so nothing is hidden in a default you did not choose.

## Citation

If you use RT, please cite the RT-J paper:

```bibtex
@article{ranjan2026rtj,
  title     = {RT-J: Large-Scale Pretraining of Relational Transformers for
               Context-Efficient Predictions},
  author    = {Ranjan, Rishabh and Kothapalli, Vignesh and Agarwal,
               Harshvardhan and Kanatsoulis, Charilaos and Upendra, Roshan and
               Palczewski, Tom and Guestrin, Carlos and Leskovec, Jure},
  year      = {2026},
}
```

[`CITATION.cff`](CITATION.cff) is the machine-readable form, and the source of
truth for the author list. To cite one of the earlier models instead, use the
paper linked in its row of the table at the top.

## License

RT is released under the [MIT License](LICENSE). The released checkpoints and
the datasets on the [`stanford-star`](https://huggingface.co/stanford-star)
HuggingFace org carry their own licenses; "the Join" is assembled from
third-party databases that keep theirs.
