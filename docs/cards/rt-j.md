---
license: cc-by-4.0
tags:
  - relational-deep-learning
  - relational-databases
  - relational-data
  - tabular
  - tabular-classification
  - tabular-regression
  - foundation-model
  - in-context-learning
  - few-shot
  - zero-shot
  - relbench
  - relational-transformer
datasets:
  - stanford-star/the-join
  - stanford-star/the-join-preprocessed
  - stanford-star/plurel
  - stanford-star/plurel-preprocessed
  - stanford-star/relbench-v1
  - stanford-star/relbench-preprocessed
metrics:
  - roc_auc
  - mae
pipeline_tag: tabular-classification
library_name: pytorch
---

# RT-J (`rt-j`)

RT-J is the released Relational Transformer checkpoint for **zero-shot /
in-context entity prediction** over multi-table relational databases. One
checkpoint handles binary entity classification and entity regression alike,
with no per-task gradient training: a task is specified by a masked target cell
and a sampled in-context neighbourhood of the relational graph, and the
prediction is a single forward pass.

- Code: <https://github.com/stanford-star/relational-transformer>
- Paper: *RT-J: Large-Scale Pretraining of Relational Transformers for
  Context-Efficient Predictions* — in progress; cite the entry below. The two
  earlier, superseded papers are
  [PluRel (arXiv:2602.04029)](https://arxiv.org/abs/2602.04029) and
  [Relational Transformer (arXiv:2510.06377)](https://arxiv.org/abs/2510.06377).
- Project page: <https://star-project.stanford.edu/rt-j>
- Docs: [downloads](https://github.com/stanford-star/relational-transformer/blob/main/docs/downloads.md) ·
  [preprocess](https://github.com/stanford-star/relational-transformer/blob/main/docs/preprocess.md) ·
  [inference](https://github.com/stanford-star/relational-transformer/blob/main/docs/inference.md) ·
  [train](https://github.com/stanford-star/relational-transformer/blob/main/docs/train.md)

## Model

| | |
|---|---|
| Parameters | 85,562,530 (~85M) |
| Blocks | 12 |
| `d_model` | 512 |
| Heads | 8 |
| `d_ff` | 2048 |
| Weight dtype | bfloat16 on disk (`from_pretrained` loads float32) |
| Text embedder | `sentence-transformers/all-MiniLM-L12-v2` (`d_text` 384) |
| Loss | Huber |

Attention is RT's relational attention over cells, rows, columns and
primary/foreign-key links. Text columns are pre-embedded by the text embedder
during preprocessing; the embedder is **not** part of this checkpoint and is not
fine-tuned.

### Files

| file | what it is |
|---|---|
| `model.safetensors` | 395 bfloat16 tensors, 171,164,252 bytes; metadata `step=9000`, `swa_n=9000` |
| `config.json` | embedder, `d_text`, and the model dims above |
| `LICENSE` | see **Licence** below |

## Usage

There is **no CLI**. RT is a library; a run is a script that calls it. Start
from [`examples/`](https://github.com/stanford-star/relational-transformer/tree/main/examples)
and edit the call.

Install `rt` — the model plus the native Rust data engine, built from source,
so you need a [Rust toolchain](https://rustup.rs) and Python 3.12+:

```bash
pip install "git+https://github.com/stanford-star/relational-transformer.git"
```

Then the whole zero-shot path in one copy-paste — download one RelBench
database already preprocessed into RT's tensor format, load these weights, and
predict whether an F1 driver fails to finish a race:

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

pre_dir = snapshot_download(
    "stanford-star/relbench-preprocessed",
    repo_type="dataset",
    allow_patterns="rel-f1/*",
)

model = RelationalTransformer.from_pretrained(
    "stanford-star/rt-j", device=device
).to(torch.bfloat16)
cfg = model.config

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

results = ev.evaluate_raw([(model, "")], [128])
_task, _ctx, _labels, out, _n = next(iter(results))
preds = torch.sigmoid(torch.tensor(out[""], dtype=torch.float32))
print("driver-dnf probability:", [round(p, 3) for p in preds.tolist()])
```

`items_per_task=5` and a 128-cell context keep this small. On a GPU it is
seconds; on a CPU it still takes tens of minutes, because `flex_attention` has
no fused bfloat16 CPU kernel. Raise `ctx_size_list` toward RT-J's training
context of 8192 (keeping `local_ctx_size <= ctx_size`) for full accuracy over a
whole test split.

To load the weights alone:

```python
from rt.model import RelationalTransformer

model = RelationalTransformer.from_pretrained("stanford-star/rt-j")
```

Reproduce the RelBench numbers end to end. Preprocessed data is downloaded up
front into a local directory (never fetched on demand); the checkpoint is still
resolved from the Hub:

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed

pixi run python examples/eval.py
```

[`examples/eval.py`](https://github.com/stanford-star/relational-transformer/blob/main/examples/eval.py)
calls `rt.eval.main` with **every argument spelled out**. `rt.eval.main` is
keyword-only and has no defaults, so it cannot be called with a short argument
list — copy the example and edit the lines you want. To run one task instead of
the 21-task list, replace

```python
        db_task_list=str(get_mixture_path("relbench", "forecast")),
```

with

```python
        db_task_list=[("rel-f1", "driver-top3")],
```

Your own database: `rt.eval` is wired to RelBench, but the pieces compose for
any database in RelBench format. The
[Colab notebook](https://github.com/stanford-star/relational-transformer/blob/main/byod/colab.ipynb)
walks through DuckDB database → tasks in SQL → preprocess → forward pass →
score, on this checkpoint.

## Training

Two phases. Phase 2 is **warm-started from the phase-1 weights**; the released
weights come from phase 2.

| | phase 1 — PluRel | phase 2 — the Join |
|---|---|---|
| data | [`stanford-star/plurel`](https://huggingface.co/datasets/stanford-star/plurel), preprocessed | [`stanford-star/the-join`](https://huggingface.co/datasets/stanford-star/the-join), preprocessed |
| databases | 1,900 synthetic | 523 real (those under the 5 GB per-database cutoff) |
| tasks | 86,211 | 13,243 forecasting and autocompletion |
| init | random | phase-1 weights |

### Context construction

For each target cell the sampler builds a context from **random-walk
retrieval** — 10,000 walks of length 20 — plus a **bounded-width local BFS**.
Four knobs are **resampled per batch**, so the model sees a spread of context
shapes rather than one:

| knob | values resampled from |
|---|---|
| context size | 512, 1024, 2048, 4096, 8192 |
| local context size | 128, 256, 512, 1024, 2048, 4096, 8192 |
| BFS width | 8, 16, 32, 64, 128, 256 |
| recency preference (`prefer_latest`) | false, true |

**Only the target cell is masked** (`mask_prob_max = 0`): there is no auxiliary
random-cell masking objective.

### Optimization

| | |
|---|---|
| Optimizer | Muon for the hidden weight matrices, AdamW for everything else |
| Learning rate | constant 5e-4 after a 2,000-step linear warmup (no decay) |
| Weight decay | 0.1 |
| Gradient clipping | global norm 1.0 |
| Global batch | 1,024 contexts |
| SWA | EMA, momentum 0.9995 |

### Checkpoint selection

The released weights are the **SWA (EMA) weights at step 9,000 of phase 2**,
selected by **mean validation AUROC (74.36)** over the RelBench forecast tasks
on their `val` splits. In-loop validation during pretraining ran on
[`stanford-star/relbench-preprocessed`](https://huggingface.co/datasets/stanford-star/relbench-preprocessed)
with the `relbench/forecast` mixture (the 21-task benchmark). Classification
AUROC is the selection metric; the same single checkpoint serves regression.

## Evaluation

RelBench leaderboard protocol: all **21 RelBench entity tasks** (12
classification, 9 regression), **full official test splits**, ensembled over the
**top-4 validation-tuned context configurations x 4 seeds** per task.

| metric | mean over tasks |
|---|---|
| AUROC (12 classification tasks) | **74.45** |
| nMAE (9 regression tasks, MAE / train-split std) | **32.35** |

The per-task tuned configurations are published as
[`reproduce/tune/tuned_configs_rt-j.json`](https://github.com/stanford-star/relational-transformer/blob/main/reproduce/tune/tuned_configs_rt-j.json):
one entry per task, each carrying `top_cfgs` — the four
`(ctx_size, local_ctx_size, bfs_width, prefer_latest)` tuples that are ensembled
— alongside `best_cfg` and the full validation grid they were selected from.
Feed a task's `top_cfgs` to `examples/eval.py` as `ctx_size_list` and
`lcs_bw_pl_grid`, or regenerate the file with the
[`reproduce/`](https://github.com/stanford-star/relational-transformer/tree/main/reproduce)
tune stage ([`reproduce/tune/README.md`](https://github.com/stanford-star/relational-transformer/blob/main/reproduce/tune/README.md)).

If you just want one good setting rather than the leaderboard protocol, use the
released default: `ctx_size` 8192, `local_ctx_size` 256, `bfs_width` 32,
`prefer_latest` true. It scores below the tuned ensemble.

### Reproducibility pins

Results depend on the preprocessed data and the sampler, both of which have
moved since. Pin the dataset revisions recorded in
[`docs/downloads.md`](https://github.com/stanford-star/relational-transformer/blob/main/docs/downloads.md)
— for RelBench, revision `1016626ddb30c027b92458bf866903850cc205e1`. An
unpinned download is not the data this model was trained on.

## Intended use

- Zero-shot / in-context prediction of a target cell in a relational database
  given a sampled context, for binary classification and scalar regression over
  entities.
- A starting point for per-task fine-tuning, and a pretrained backbone for
  research on relational foundation models.

## Limitations and out-of-scope use

- **Binary classification and scalar regression over entities only.** Not
  multiclass, not link prediction, not recommendation, not generation.
- Input must be in RelBench format and preprocessed by `rustler`; nothing is
  fetched on demand, so a `pre_dir` is always a local directory.
- Text is consumed only through frozen `all-MiniLM-L12-v2` embeddings — long or
  nuanced text is heavily compressed, and a different embedder does not match
  this checkpoint. The preprocessed repositories ship embeddings and no
  readable strings, so switching embedder means re-running preprocessing from
  the raw collection, not re-embedding the preprocessed tree.
- Accuracy depends strongly on the context configuration; the leaderboard
  numbers above use per-task tuned configurations and ensembling, and a single
  default-context forward pass scores lower.
- Pretrained partly on synthetic databases and on a crawl of public databases.
  It carries whatever biases and errors those contain, has no calibration
  guarantees, and must not be used unexamined for decisions about people.
- No guarantee that an evaluation database was absent from pretraining beyond
  the decontamination applied to the Join (databases overlapping RelBench
  sources were dropped); check for yourself before claiming zero-shot on a new
  dataset.

## Licence

These weights are released under **CC BY 4.0** (see `LICENSE`) — attribution
only, commercial use permitted. This replaces the CC BY-NC-SA 4.0 licence the
repository previously carried.

The licence covers **the released weights**, which are what this repository
distributes. It does not extend to the training data, which carries its own,
stricter terms: the Join is assembled from third-party databases that keep
their own licences, and
[`stanford-star/the-join`](https://huggingface.co/datasets/stanford-star/the-join)
is **CC BY-SA 4.0** with per-database attribution in
`STATS/databases.parquet`. If you redistribute any of that data, or a
derivative of it, those terms govern it — not this one.

RT's own source code is MIT; see the
[repository](https://github.com/stanford-star/relational-transformer/blob/main/LICENSE).

## Related

- Datasets: [the-join](https://huggingface.co/datasets/stanford-star/the-join) ·
  [the-join-preprocessed](https://huggingface.co/datasets/stanford-star/the-join-preprocessed) ·
  [plurel](https://huggingface.co/datasets/stanford-star/plurel) ·
  [plurel-preprocessed](https://huggingface.co/datasets/stanford-star/plurel-preprocessed) ·
  [relbench](https://huggingface.co/datasets/stanford-star/relbench-v1) ·
  [relbench-preprocessed](https://huggingface.co/datasets/stanford-star/relbench-preprocessed)
- Models: [rt-plurel](https://huggingface.co/stanford-star/rt-plurel) ·
  [rt-v1](https://huggingface.co/stanford-star/rt-v1)

## Citation

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
