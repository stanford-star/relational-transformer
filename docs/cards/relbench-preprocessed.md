---
# CC BY 4.0 covers these artifacts because they contain no verbatim upstream
# content: the source strings are not shipped, only embeddings of them. The
# "Licence" section below says what that does and does not cover.
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
| Files | 103 |
| Size | 208,537,747,984 bytes (~194.2 GiB) |
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
legacy/<db>/          # the same artifacts for the legacy RT-v1 code path
```

Two things `rustler` writes during preprocessing are **not shipped**, because
nothing reads them afterwards and shipping them would put verbatim source
content in this repository. `text.json`, the deduplicated table of source
strings the embedder consumes, is one: a string cell in `nodes.rkyv` is an
index, and training and inference gather the embedding at that index straight
out of `text_emb_*.bin`. `legacy/_transformed/`, the RelBench-format parquet the
RT-v1 transform emits on its way into `rustler`, is the other; it is an
intermediate, the legacy path reads `legacy/<db>/`, and
`rt.preprocess.legacy` regenerates it from
[`stanford-star/relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1)
whenever it is actually wanted. The consequence for you: to use a different
text embedder, re-run preprocessing from the raw repository rather than
re-embedding this tree.

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

### Column statistics

`rustler` z-scores numeric and datetime cells. Since PR #4 (`rustler: column
stats from the train period only`, commit `8030aa8`) the statistics come from
the train period alone, where previously they were taken over the whole table.
34 of the 50 database tables here are time-indexed and so take narrower
statistics under that change.

The tree published at this revision predates `8030aa8`; a regenerated one is on
its way and will be uploaded here, with the preprocessing commit above updated
to match. Pin the revision below for the published numbers.

## Revisions

The RT-J paper's results were produced against revision
**`1016626ddb30c027b92458bf866903850cc205e1`** (2026-08-08). Pin it:

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --revision 1016626ddb30c027b92458bf866903850cc205e1 \
  --local-dir data/relbench-preprocessed
```

## Licence

**CC BY 4.0** — attribution only, commercial use permitted.

The raw collection,
[`stanford-star/relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1),
is **CC BY-SA 4.0** and stays that way: all seven databases are declared
share-alike there, and at least `rel-stack` inherits that genuinely rather than
by choice, being built from the
[Stack Exchange data dumps](https://archive.org/details/stackexchange), whose
user-contributed content is CC BY-SA 4.0.

This repository can be more permissive because **it contains no verbatim
upstream content.** Per database it holds, and only holds:

| | |
|---|---|
| `text_emb_*.bin` | one 384-d `bf16` MiniLM embedding per distinct source string |
| `nodes.rkyv` | numeric and datetime cells z-scored, text cells as an embedding index, raw row timestamps |
| `offsets.rkyv`, `p2f_adj.rkyv` | row offsets and the foreign-key graph |
| `meta.json`, `table_info.json`, `column_index.json` | table and column names, row counts, semantic types |

Neither the string cell values nor the RelBench-format parquet are here — see
**File layout** for what is left out and why. No string or free-text column can
be read back out, the z-scoring statistics are not serialized so numeric scale
is unrecoverable, and primary-key columns are dropped. What remains is a lossy
derived representation plus schema metadata — the same basis on which the
released checkpoints are CC BY 4.0.

So CC BY 4.0 covers **these artifacts**, not the underlying databases and not
the benchmark. To work with the source data, go to
[`stanford-star/relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1)
and follow its terms. Attribution for the databases belongs to RelBench and to
each upstream source: cite RelBench, not only this repository.

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
