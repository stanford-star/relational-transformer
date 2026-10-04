---
license: cc-by-4.0
pretty_name: RelBench (preprocessed for Relational Transformer)
tags:
  - relational-deep-learning
  - relational-databases
  - relational-data
  - tabular
  - benchmark
  - relbench
  - relational-transformer
task_categories:
  - tabular-classification
  - tabular-regression
size_categories:
  - 100B<n<1T
# No `configs:` block: binary rustler artifacts, not viewer-readable tables.
---

# RelBench, preprocessed (`relbench-preprocessed`)

The evaluation and in-loop-validation data for the Relational Transformer: the
seven [RelBench](https://relbench.stanford.edu) databases run through the
`rustler` preprocessor, in the on-disk format the RT dataloaders read. This is
the `pre_dir` that reproduces the published RT-J numbers.

- Code: <https://github.com/stanford-star/relational-transformer>
- Paper: [arXiv:2510.06377](https://arxiv.org/abs/2510.06377)
- Download / inference:
  [`docs/downloads.md`](https://github.com/stanford-star/relational-transformer/blob/main/docs/downloads.md) ·
  [`docs/inference.md`](https://github.com/stanford-star/relational-transformer/blob/main/docs/inference.md)

## What it is

| | |
|---|---|
| Databases | 7 (`rel-amazon`, `rel-avito`, `rel-event`, `rel-f1`, `rel-hm`, `rel-stack`, `rel-trial`) |
| Tasks | 34 total — 21 forecasting (the RelBench entity benchmark), 13 autocompletion |
| Files | 293 |
| Size | 244,050,090,877 bytes (~227.3 GiB) |
| Text embedder | `sentence-transformers/all-MiniLM-L12-v2` (384-d) |

## Relation to the raw dataset

Derived from [`stanford-star/relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1),
the RelBench databases and task tables in RelBench format (parquet +
`manifest.yaml`), which also carries the regression-target standard deviations
that turn MAE into nMAE. That repository is the one to cite and the one to
re-preprocess from; this one is a build artifact.

## File layout

```
<db>/                 # one per database
  meta.json  table_info.json  column_index.json
  nodes.rkyv  offsets.rkyv  p2f_adj.rkyv
  text_emb_all-MiniLM-L12-v2.bin
  text.json
legacy/<db>/          # the same artifacts for the legacy RT-v1 code path
legacy/_transformed/  # RelBench-format parquet the RT-v1 transform emits
```

The task lists are **not** in this repository. The curated (db, task) mixtures
are vendored in the `rt` package instead, so a list can no longer drift from
the data or the code that reads it:

```python
from rt.data import get_mixture_path, list_mixtures

list_mixtures()                                    # every (collection, name)
path = get_mixture_path("relbench", "forecast")
```

Pass that path as `db_task_list`.

For this collection the vendored lists are `all` (34 pairs over the 7
databases), `forecast` (21 — the RelBench entity benchmark, the default
evaluation list) and `autocomplete` (13).

## How to load it

`pre_dir` is always a **local directory**; nothing is fetched on demand.
Download it, then point a run at the path:

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed
pixi run python examples/eval.py        # pre_dir="data/relbench-preprocessed"
```

There is no CLI; `examples/eval.py` is a script that calls `rt.eval.main` with
every argument spelled out, and the released evaluation values live in it. The
checkpoint is the one thing still resolved from the Hub on demand, so
`load_ckpt_path="stanford-star/rt-j"` needs no download.

## How it was produced

```python
# examples/preprocess.py
one(dataset="stanford-star/relbench-v1/rel-f1", out_dir="data/relbench-preprocessed",
    embedder="all-MiniLM-L12-v2", ...)   # once per database
```

There is no CLI: copy
[`examples/preprocess.py`](https://github.com/stanford-star/relational-transformer/blob/main/examples/preprocess.py),
edit the call, `pixi run python examples/preprocess.py`.

## Revisions

The RT-J paper's results were produced against revision
**`1016626ddb30c027b92458bf866903850cc205e1`** (2026-08-08), which predates the
rebuild. Hub revisions are immutable, so that pin still resolves to exactly the
data the paper used; pin it to reproduce published numbers:

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --revision 1016626ddb30c027b92458bf866903850cc205e1 \
  --local-dir data/relbench-preprocessed
```

The rebuilt tree landed at revision
**`ff29544d3622294d579a05afcdda21b0906283b2`** (2026-10-01); later revisions
change only this card.

## Licence

**CC BY 4.0** — attribution only, commercial use permitted.

Cite the RT-J paper below, and RelBench for the benchmark. The raw collection
is
[`stanford-star/relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1).

## Citation

Cite RelBench for the benchmark and the RT paper for this preprocessing:

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
