---
license: cc-by-4.0
pretty_name: The Join (preprocessed for Relational Transformer)
tags:
  - relational-deep-learning
  - relational-databases
  - relational-data
  - tabular
  - pretraining-data
  - relational-transformer
task_categories:
  - tabular-classification
  - tabular-regression
size_categories:
  - 100B<n<1T
# No `configs:` block on purpose: this repository holds binary rustler
# artifacts, not parquet/csv the dataset viewer can read.
---

# The Join, preprocessed (`the-join-preprocessed`)

The pretraining corpus of [RT-J](https://huggingface.co/stanford-star/rt-j), in
the on-disk format the Relational Transformer dataloaders read. It is
[`stanford-star/the-join`](https://huggingface.co/datasets/stanford-star/the-join)
— real-world multi-table databases in RelBench format — run through the
`rustler` preprocessor, plus frozen text-column embeddings.

- Code: <https://github.com/stanford-star/relational-transformer>
- Paper: [arXiv:2510.06377](https://arxiv.org/abs/2510.06377)
- How to download a subset:
  [`docs/downloads.md`](https://github.com/stanford-star/relational-transformer/blob/main/docs/downloads.md)
- How it is produced:
  [`docs/preprocess.md`](https://github.com/stanford-star/relational-transformer/blob/main/docs/preprocess.md)

## What it is

| | |
|---|---|
| Databases | 523 (those under the 5 GB per-database cutoff, `all_5gb_cutoff`) |
| Tasks | 13,243 (db, task) pairs over those 523 databases |
| Files | 4,186 |
| Size | 278,792,483,910 bytes (~259.6 GiB) |
| Text embedder | `sentence-transformers/all-MiniLM-L12-v2` (384-d) |

Per-file-kind totals, which is what a partial download is planned against:

| file | total |
|---|---|
| `nodes.rkyv` | ~195.1 GiB |
| `text_emb_all-MiniLM-L12-v2.bin` | ~29.8 GiB |
| `p2f_adj.rkyv` | ~26.1 GiB |
| `offsets.rkyv` | ~5.3 GiB |

## Relation to the raw dataset

`stanford-star/the-join` is the raw side: parquet tables plus a
`manifest.yaml` of relational metadata, per database, in RelBench format. This
repository is derived from it and is **not** a substitute — the raw repository
is what you need to re-run preprocessing, and it carries the per-database
licence attribution.

The RT-J paper describes the collection at **639 databases**, which is the
untrimmed raw side. This preprocessed repository was **trimmed on 2026-09-12 to
the 523 `all_5gb_cutoff` databases** (116 dropped), and those 523 are the set
every published RT-J number was pretrained on. A reader who pinned an earlier
revision gets a different, larger dataset; see **Revisions** below.

## File layout

One directory per database:

```
<db>/
  meta.json                            # relational + task metadata rustler emits
  table_info.json                      # table names, row counts
  column_index.json                    # column -> index, semantic type
  nodes.rkyv                           # rkyv-serialized cell values (the bulk)
  offsets.rkyv                         # row offsets into nodes
  p2f_adj.rkyv                         # primary->foreign key adjacency
  text_emb_all-MiniLM-L12-v2.bin       # frozen text-column embeddings
  text.json                            # the interned source strings
```

The task lists are **not** in this repository. The curated (db, task) mixtures
are vendored in the `rt` package instead, so a list can no longer drift from
the data or the code that reads it:

```python
from rt.data import get_mixture_path, list_mixtures

list_mixtures()                                    # every (collection, name)
path = get_mixture_path("the-join", "rt-j")
```

Pass that path as `db_task_list`.

For this collection the vendored lists are: `rt-j` and `all` (both 13,243 pairs
over the 523 databases — they are the same list, and `rt-j` is RT-J's phase-2
pretraining mixture), `autocomplete` (9,145 over 519) and `forecast` (4,098 over
153).

## How to load it

`pre_dir` is always a **local directory** — nothing is fetched on demand.
Download it, then point a run at the path:

```bash
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed
pixi run python examples/train.py        # pre_dir="data/the-join-preprocessed"
```

There is no CLI; `examples/train.py` is a script that calls `rt.train.main` with
every argument spelled out, and the released pretraining values live in it. Its
`db_task_list` comes from `rt.data.get_mixture_path("the-join", "rt-j")`.
[`docs/downloads.md`](https://github.com/stanford-star/relational-transformer/blob/main/docs/downloads.md)
has the `--include` patterns that skip the embedders you are not using, and the
revision pins below.

## How it was produced

```python
# examples/preprocess.py, preprocess_a_collection()
many(repo="stanford-star/the-join", out_dir="data/the-join-preprocessed",
     shard=0, num_shards=1, skip_existing=True, embedder="all-MiniLM-L12-v2", ...)
```

There is no CLI: copy
[`examples/preprocess.py`](https://github.com/stanford-star/relational-transformer/blob/main/examples/preprocess.py),
edit the call, `pixi run python examples/preprocess.py`. `shard=i,
num_shards=N` splits the collection across a job array.

**Preprocessing commit.** The content of this repository was written by
`rustler` at repository commit **`dc56007`** (2026-08-05, the last
preprocessing-code commit before the upload on 2026-08-05/07); the 2026-09-12
commits are a trim and a task-list regeneration, not a re-preprocessing.

## Revisions

The RT-J paper's results were produced against revision
**`ba5574ba659f0ef8b592793ed2ac9cb78dc87100`** (2026-09-12, the completed trim).
Pin it:

```bash
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --revision ba5574ba659f0ef8b592793ed2ac9cb78dc87100 \
  --local-dir data/the-join-preprocessed
```

## Licence

**CC BY 4.0** — attribution only, commercial use permitted.

Cite the RT-J paper below. For the databases themselves, the raw collection is
[`stanford-star/the-join`](https://huggingface.co/datasets/stanford-star/the-join),
whose `STATS/databases.parquet` carries the per-database `source_url` for
attribution.

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
