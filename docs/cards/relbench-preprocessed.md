---
# CC BY-SA 4.0 is inherited from stanford-star/relbench-v1, whose sources include
# the Stack Exchange dumps. The "Licence" section below says so in prose.
license: cc-by-sa-4.0
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
| Files | 295 |
| Size | 244,050,085,127 bytes (~227.3 GiB) |
| Text embedder | `sentence-transformers/all-MiniLM-L12-v2` (384-d) |

## Relation to the raw dataset

Derived from [`stanford-star/relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1),
the RelBench databases and task tables in RelBench format (parquet +
`manifest.yaml`), which also carries the regression-target standard deviations
that turn MAE into nMAE. That repository is the one to cite and the one to
re-preprocess from; this one is a build artifact.

## File layout

```
db-task-lists/
  all.json            # 34 (db, task) pairs
  forecast.json       # the 21-task RelBench entity benchmark -- the default eval list
  autocomplete.json   # 13 cell-autocompletion tasks
<db>/                 # one per database
  meta.json  table_info.json  column_index.json
  nodes.rkyv  offsets.rkyv  p2f_adj.rkyv
  text_emb_all-MiniLM-L12-v2.bin
  text.json
legacy/_transformed/  # RelBench-format parquet produced by the RT-v1 transform,
                      # kept for the legacy RT-v1 code path; not needed by RT-J
```

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

**Preprocessing commit.** Written by `rustler` at repository commit
**`cc562b6`** (2026-08-07, the last preprocessing-code commit before the
2026-08-08 upload).

### This revision carries pre-fix, leaky normalization

`rustler` PR #4 (`rustler: column stats from the train period only`, commit
`8030aa8`, merged 2026-10-01) is a temporal-leakage fix: before it, per-column
z-scoring statistics and the global datetime statistics were computed over the
validation and test rows as well as the train period, so training and inference
inputs were normalized with statistics that had seen the future. The on-disk
format did not change, so **this dataset keeps loading, and keeps the old leaky
normalization, until it is regenerated.** A regenerated copy is in preparation
and will land as a new revision; the revision pinned below will remain the one
the published numbers were produced against.

How much of the data moves, measured against this revision: **34 of the 50
database tables are time-indexed** and so take narrower statistics under the
fix, and **27 of 40 numeric columns shift by more than 0.1 of their old standard
deviation**, the largest by 1.43. This is a real change to the inputs, not a
rounding difference.

How much the *results* move: measured with the released
[`rt-j`](https://huggingface.co/stanford-star/rt-j) checkpoint at its released
default context (`ctx_size` 8192, `local_ctx_size` 256, `bfs_width` 32,
`prefer_latest`), the same commit and seed on both arms, one context seed, and
only `pre_dir` differing between this published tree and a regenerated one:

| task | metric | published | regenerated | change | n |
|---|---|---:|---:|---:|---:|
| `rel-f1/driver-dnf` | AUROC ↑ | 82.843 | 82.537 | −0.306 | 702 |
| `rel-f1/driver-top3` | AUROC ↑ | 90.537 | 90.683 | +0.146 | 726 |
| `rel-event/user-repeat` | AUROC ↑ | 79.101 | 79.088 | −0.013 | 246 |
| `rel-hm/user-churn` | AUROC ↑ | 61.955 | 61.947 | −0.008 | 4096 |
| `rel-f1/driver-position` | nMAE ↓ | 38.667 | 39.107 | +0.440 | 760 |
| `rel-trial/study-adverse` | nMAE ↓ | 16.413 | 16.375 | −0.038 | 3098 |
| `rel-avito/ad-ctr` | nMAE ↓ | 42.624 | 42.117 | −0.507 | 1816 |
| **mean AUROC** (4 tasks) | ↑ | 78.609 | 78.564 | **−0.045** | |
| **mean nMAE** (3 tasks) | ↓ | 32.568 | 32.533 | **−0.035** | |

No task moves by more than 0.51; five of the seven move by less than 0.31 and
two by less than 0.02. The sign is mixed and the differences largely cancel,
which is what you expect from a leak in *input normalization* rather than in
labels.

Read this as a bound on the magnitude, not as a correction to any published
number. It is one context seed over at most 4,096 evaluation rows per task on
seven tasks, with the released checkpoint rather than a retrained one, and it
does not re-derive the paper's tables.

> **The `legacy/` tree moves too, and the RT-v1 checkpoints are not covered by
> that measurement.** `legacy/_transformed/` feeds the RT-v1 code path, and it
> changes under the fix as well, so the released RT-v1 checkpoints are orphaned
> from the current preprocessor in the same way — and the six-task measurement
> above was run with `rt-j`, not with them. Nobody has measured the effect on
> RT-v1. Pin the revision below to reproduce RT-v1 results.

## Revisions

The RT-J paper's results were produced against revision
**`1016626ddb30c027b92458bf866903850cc205e1`** (2026-08-08). Pin it:

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --revision 1016626ddb30c027b92458bf866903850cc205e1 \
  --local-dir data/relbench-preprocessed
```

## Licence

**CC BY-SA 4.0**, inherited from
[`stanford-star/relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1)
and, through it, from upstream.

All seven databases are declared CC BY-SA 4.0 in that repository's
`STATS/databases.parquet` — `rel-amazon`, `rel-avito`, `rel-event`, `rel-f1`,
`rel-hm`, `rel-stack`, `rel-trial` — and at least one of them inherits that
genuinely rather than by RelBench's choice: `rel-stack` is built from the
[Stack Exchange data dumps](https://archive.org/details/stackexchange), which
are themselves CC BY-SA 4.0 user-contributed content. One such source is enough
to carry share-alike onto the aggregate. (That blanket per-row declaration is
RelBench's, not a per-source re-verification by us; some sources, such as
clinicaltrials.gov behind `rel-trial`, are upstream public domain.)

**Why the obligation carries into the preprocessed form.** This is a format
conversion, not an independent work. `text.json` is a verbatim, deduplicated
intern table of every string cell value in the source database and `nodes.rkyv`
stores the index into it, so every string and free-text column is exactly
reconstructible — for `rel-stack` that means Stack Exchange post text,
unchanged. Numeric and datetime cells are z-scored without the statistics
needed to invert them, and primary keys are dropped, but that does not make the
textual content derived.

What share-alike does and does not do here: it applies when you redistribute
this dataset or a modified version of it. It does **not** restrict commercial
use, and it does **not** reach model weights trained on the data — the
checkpoints released alongside it are CC BY 4.0.

Attribution for the underlying databases belongs to RelBench and to each
upstream source; cite RelBench, not only this repository.

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
