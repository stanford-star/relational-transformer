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
| Files | 16,002 |
| Size | 43,207,210,158 bytes (~40.2 GiB) |
| Text embedder | `sentence-transformers/all-MiniLM-L12-v2` (384-d) |

Per-file-kind totals: `nodes.rkyv` ~35.2 GiB, `p2f_adj.rkyv` ~4.1 GiB,
`offsets.rkyv` ~0.8 GiB, `text_emb_all-MiniLM-L12-v2.bin` ~0.1 GiB (synthetic
databases carry little text).

The curated pretraining mixture is `plurel/rt-plurel-train.json`, vendored in
the `rt` package:
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
plurel-<n>/              # one per synthetic database, n = 3000..4999
  meta.json  table_info.json  column_index.json
  nodes.rkyv  offsets.rkyv  p2f_adj.rkyv
  text_emb_all-MiniLM-L12-v2.bin
  text.json
```

The task lists are **not** in this repository. The curated (db, task) mixtures
are vendored in the `rt` package instead, so a list can no longer drift from
the data or the code that reads it:

```python
from rt.data import get_mixture_path, list_mixtures

list_mixtures()                                    # every (collection, name)
path = get_mixture_path("plurel", "rt-plurel-train")
```

Pass that path as `db_task_list`.

For this collection the vendored lists are `rt-plurel-train` (86,211 pairs over
1,900 databases — the mixture RT-J phase 1 and RT-PluRel train on), `all` and
`autocomplete` (116,088 over 1,973 each; every task here is an autocompletion
task, so `forecast` is empty).

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

The `rt-plurel-train` mixture is vendored in the package, so you do not have to
rebuild it, but the filter that produced it is worth knowing because it is what
separates the 1,900 trained-on databases from the 2,000 present. Taking
`plurel-3000` … `plurel-4999` in order, a **database** is dropped if any of its
tables has more than 5 foreign keys, and the first 1,900 survivors are kept. A
**column** of a surviving database becomes a task only if it is a `bool`,
`int64` or `double` column whose name contains `feature`, it is not a source-node
column, and it has at least 2 distinct values — then, for a classification task
(the `bool` columns), at least 2 classes and a majority class no larger than
99%; for a regression task, a standard deviation of at least 1e-4. That yields
the 86,211 pairs in the vendored list.

## Revisions

The RT-J paper's results were produced against revision
**`9d70172425b44053f1270b19094ba6dcf7d66464`** (2026-09-12). Hub revisions are
immutable, so pin it to reproduce published numbers:

```bash
pixi run hf download stanford-star/plurel-preprocessed --repo-type dataset \
  --revision 9d70172425b44053f1270b19094ba6dcf7d66464 \
  --local-dir data/plurel-preprocessed
```

The current data landed at revision
**`e18c60c9feff5d26d3440cae2d8d92b8adb2a620`** (2026-10-01); later revisions
change only this card.

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

[`the-join-preprocessed`](https://huggingface.co/datasets/stanford-star/the-join-preprocessed)
and
[`relbench-preprocessed`](https://huggingface.co/datasets/stanford-star/relbench-preprocessed)
are CC BY 4.0 as well.

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
