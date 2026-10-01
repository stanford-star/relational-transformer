---
license: cc-by-4.0
pretty_name: PluRel (preprocessed for Relational Transformer)
tags:
  - relational-deep-learning
  - relational-databases
  - relational-data
  - tabular
  - synthetic-data
  - pretraining-data
  - relational-transformer
task_categories:
  - tabular-classification
  - tabular-regression
size_categories:
  - 10B<n<100B
# No `configs:` block: binary rustler artifacts, not viewer-readable tables.
---

# PluRel, preprocessed (`plurel-preprocessed`)

Phase 1 of [RT-J](https://huggingface.co/stanford-star/rt-j)'s pretraining, and
the pretraining corpus of
[RT-PluRel](https://huggingface.co/stanford-star/rt-plurel): the synthetic
relational databases of
[`stanford-star/plurel`](https://huggingface.co/datasets/stanford-star/plurel)
run through the `rustler` preprocessor, in the on-disk format the Relational
Transformer dataloaders read.

- Code: <https://github.com/stanford-star/relational-transformer>
- Papers: [Relational Transformer (arXiv:2510.06377)](https://arxiv.org/abs/2510.06377) ·
  [PluRel (arXiv:2602.04029)](https://arxiv.org/abs/2602.04029)

## What it is

| | |
|---|---|
| Databases | 2,000 synthetic (`plurel-3000` … `plurel-4999`) |
| Files | 16,005 |
| Size | 43,218,968,510 bytes (~40.2 GiB) |
| Text embedder | `sentence-transformers/all-MiniLM-L12-v2` (384-d) |

Per-file-kind totals: `nodes.rkyv` ~35.2 GiB, `p2f_adj.rkyv` ~4.1 GiB,
`offsets.rkyv` ~0.8 GiB, `text_emb_all-MiniLM-L12-v2.bin` ~0.1 GiB (synthetic
databases carry little text).

The curated pretraining mixture is `db-task-lists/rt-plurel-train.json`:
**86,211 (db, task) pairs over 1,900 of the 2,000 databases** — the filtered
subset RT-J's phase 1 and RT-PluRel actually train on. The remaining 100
databases are present but excluded by the filter.

## Relation to the raw dataset

Derived from [`stanford-star/plurel`](https://huggingface.co/datasets/stanford-star/plurel)
— synthetic databases in RelBench format (parquet + `manifest.yaml`), generated
by PluRel. That repository is the one to generate from or cite; this one is a
build artifact and is not a substitute.

A **2026-09-12** commit dropped the older generation of databases, leaving only
`plurel-3000` … `plurel-4999`; the same round regenerated the task lists with
autocomplete task manifests and the filtered `rt-plurel-train.json`. An earlier
revision is a different collection.

## File layout

```
db-task-lists/
  all.json
  forecast.json
  autocomplete.json
  rt-plurel-train.json   # the curated 86,211-task / 1,900-db training mixture
plurel-<n>/              # one per synthetic database, n = 3000..4999
  meta.json  table_info.json  column_index.json
  nodes.rkyv  offsets.rkyv  p2f_adj.rkyv
  text_emb_all-MiniLM-L12-v2.bin
  text.json
```

`pre_dir` is always a **local directory**; download with
`hf download --local-dir`, nothing is fetched on demand.

## How it was produced

```python
# examples/preprocess.py, preprocess_a_collection()
many(repo="stanford-star/plurel", out_dir="data/plurel-preprocessed",
     shard=0, num_shards=1, skip_existing=True, embedder="all-MiniLM-L12-v2", ...)
```

There is no CLI: copy
[`examples/preprocess.py`](https://github.com/stanford-star/relational-transformer/blob/main/examples/preprocess.py),
edit the call, `pixi run python examples/preprocess.py`.

`db-task-lists/rt-plurel-train.json` ships with the data, so you do not have to
rebuild it, but the filter that produced it is worth knowing because it is what
separates the 1,900 trained-on databases from the 2,000 present. Taking
`plurel-3000` … `plurel-4999` in order, a **database** is dropped if any of its
tables has more than 5 foreign keys, and the first 1,900 survivors are kept. A
**column** of a surviving database becomes a task only if it is a `bool`,
`int64` or `double` column whose name contains `feature`, it is not a source-node
column, and it has at least 2 distinct values — then, for a classification task
(the `bool` columns), at least 2 classes and a majority class no larger than
99%; for a regression task, a standard deviation of at least 1e-4. That yields
the 86,211 pairs in the shipped list.

**Preprocessing commit.** The 2026-09-12 content was written by `rustler` at
repository commit **`970167c`** (2026-09-09, `rustler: no boolean sem type in
sampler output; bools are z-scored numbers end to end`), the last
preprocessing-code commit before that rebuild.

> **This is pre-fix, leaky normalization.** `rustler` PR #4 (`rustler: column
> stats from the train period only`, commit `8030aa8`, merged 2026-10-01)
> restricted z-scoring statistics to the train period; before it, column
> statistics and the global datetime statistics were computed over the
> validation and test rows too. The on-disk format is unchanged, so this
> dataset keeps the **old, leaky** normalization until it is regenerated. RT-J
> phase 1 and RT-PluRel were both trained on this pre-#4 data. The fix engages
> here because PluRel manifests carry a real `val_timestamp`; it does *not*
> engage on
> [`the-join-preprocessed`](https://huggingface.co/datasets/stanford-star/the-join-preprocessed),
> whose manifests have `val_timestamp: null`. For a measured bound on how far
> the fix moves downstream numbers, see the
> [`relbench-preprocessed`](https://huggingface.co/datasets/stanford-star/relbench-preprocessed)
> card: mean −0.045 AUROC and −0.035 nMAE over seven paired tasks, mixed in
> sign, one context seed. That bounds the magnitude; it does not correct any
> published number.

## Revisions

The RT-J paper's results were produced against revision
**`9d70172425b44053f1270b19094ba6dcf7d66464`** (2026-09-12). Pin it:

```bash
pixi run hf download stanford-star/plurel-preprocessed --repo-type dataset \
  --revision 9d70172425b44053f1270b19094ba6dcf7d66464 \
  --local-dir data/plurel-preprocessed
```

## Licence

**CC BY 4.0** — attribution only, commercial use permitted.

Note the asymmetry with the upstream collection, which is deliberate. The
databases here are **entirely synthetic**: they were generated by
[PluRel](https://github.com/stanford-star/plurel), contain no third-party data,
and carry no inherited terms. `stanford-star` holds the rights in both the
upstream collection and this derived artifact, so this repository can be
offered under the more permissive CC BY 4.0 even though
[`stanford-star/plurel`](https://huggingface.co/datasets/stanford-star/plurel)
is published as CC BY-SA 4.0. The upstream collection keeps its own CC BY-SA
4.0 terms when you redistribute *it*; CC BY 4.0 here applies to this
preprocessed build.

This reasoning does not carry over to the preprocessed builds of real-world
data (`the-join-preprocessed`, `relbench-preprocessed`), whose source
databases are third-party and share-alike.

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
