---
# CC BY 4.0 covers these artifacts because they contain no verbatim upstream
# content: the source strings are not shipped, only embeddings of them. The
# "Licence" section below says what that does and does not cover.
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
| Files | 3,667 |
| Size | 275,371,755,983 bytes (~256.5 GiB) |
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
untrimmed raw side; this preprocessed repository holds the 523 of them that fall
under the 5 GB per-database cutoff, and that is the set every published RT-J
number was pretrained on.

The raw collection holds 639 databases; this preprocessed repository was
**trimmed on 2026-09-12 to the 523 `all_5gb_cutoff` databases** (116 dropped).
A reader who pinned an earlier revision gets a different, larger dataset. See
**Revisions** below.

## File layout

One directory per database, plus shared task lists:

```
db-task-lists/
  all.json            # every (db, task) pair
  forecast.json       # temporal forecasting tasks
  autocomplete.json   # cell-autocompletion tasks
  rt-j.json           # the mixture RT-J's pretraining example points at
<db>/
  meta.json                            # relational + task metadata rustler emits
  table_info.json                      # table names, row counts
  column_index.json                    # column -> index, semantic type
  nodes.rkyv                           # rkyv-serialized cell values (the bulk)
  offsets.rkyv                         # row offsets into nodes
  p2f_adj.rkyv                         # primary->foreign key adjacency
  text_emb_all-MiniLM-L12-v2.bin       # frozen text-column embeddings
```

There is no `text.json`. `rustler` writes one during preprocessing — the
deduplicated table of source strings that the embedder consumes — and it is
**not shipped**, because nothing reads it after the embeddings exist: a string
cell in `nodes.rkyv` is an index, and training and inference gather the
embedding at that index straight out of `text_emb_*.bin`. Removing it is also
what makes the licence below true. The consequence for you: to use a different
text embedder, re-run preprocessing from the raw repository rather than
re-embedding this tree.

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
`db_task_list` points at `db-task-lists/rt-j.json`, which is stale at this
revision — see **Known defects** below for the one-block fix.
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

### Column statistics

`rustler` z-scores numeric and datetime cells. Since PR #4 (`rustler: column
stats from the train period only`, commit `8030aa8`) the statistics come from
the train period alone. That change leaves this collection byte-identical:
every manifest in `stanford-star/the-join` has `val_timestamp: null` and the
collection has no validation or test splits, so neither half of the fix
engages.

PR #3 (`rustler: draw BFS children without rejection sampling`, `fly.rs`)
changed the context *sampler*, not this data.

## Known defects at the current revision

- **The `db-task-lists/` files are stale with respect to the 2026-09-12 trim.**
  They name databases that are no longer present: `all.json` 16,208 pairs over
  590 dbs (67 missing, 2,965 unusable pairs), `autocomplete.json` 10,185 over
  585 (66 missing), `forecast.json` 6,023 over 204 (51 missing), and
  `rt-j.json` 10,815 over 469 (26 missing, 911 unusable pairs).
- Consequently **`rt-j.json` as shipped is not RT-J's pretraining mixture.** The
  paper's phase-2 mixture is **13,243 (db, task) pairs over exactly the 523
  databases present here**, and `rt-j.json` resolves to 9,904.

`get_tasks` reports and then ignores a task whose database is absent rather than
failing, so an unmodified run gets a silently smaller mixture than the file
names. Until the lists are regenerated, filter any of them to the databases that
are actually present. For the RT-J mixture specifically, that is all it takes —
`all.json` restricted to the present databases **is** the paper's mixture, pair
for pair:

```python
import json, os

pre_dir = "data/the-join-preprocessed"
present = {d for d in os.listdir(pre_dir) if d != "db-task-lists"}
pairs = json.load(open(f"{pre_dir}/db-task-lists/all.json"))
mixture = [p for p in pairs if p[0] in present]
assert len(mixture) == 13_243          # 16,208 named, 2,965 dropped as absent
json.dump(mixture, open(f"{pre_dir}/db-task-lists/rt-j-fixed.json", "w"))
```

Then point `db_task_list` at `rt-j-fixed.json` instead of `rt-j.json`.

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

The raw collection,
[`stanford-star/the-join`](https://huggingface.co/datasets/stanford-star/the-join),
is **CC BY-SA 4.0** and stays that way: it is an aggregate of 639 third-party
databases, about half of them share-alike upstream (Spider 1.0 143, BIRD 58,
Stack Exchange dumps 40, Wikipedia-derived 5, ODbL 4, CC BY-SA 3.0 3, GPL 2.0
1), and that obligation is inherited rather than chosen. Per-database
attribution — the `license` and `source_url` columns for all 639 — is in
`STATS/databases.parquet` there.

This repository can be more permissive because **it contains no verbatim
upstream content.** Per database it holds, and only holds:

| | |
|---|---|
| `text_emb_*.bin` | one 384-d `bf16` MiniLM embedding per distinct source string |
| `nodes.rkyv` | numeric and datetime cells z-scored, text cells as an embedding index, raw row timestamps |
| `offsets.rkyv`, `p2f_adj.rkyv` | row offsets and the foreign-key graph |
| `meta.json`, `table_info.json`, `column_index.json` | table and column names, row counts, semantic types |

The string cell values themselves are not here: `text.json` is a preprocessing
intermediate and is not shipped, so no string or free-text column can be read
back out. The z-scoring statistics are not serialized either, so numeric scale
is unrecoverable, and primary-key columns are dropped. What remains is a lossy
derived representation plus schema metadata — the same basis on which the
released checkpoints are CC BY 4.0.

So CC BY 4.0 covers **these artifacts**. It is not a licence for the underlying
databases, and it is not a route around their terms:

- To work with the source data, go to
  [`stanford-star/the-join`](https://huggingface.co/datasets/stanford-star/the-join)
  and follow each database's own licence.
- Cite the RT-J paper for this artifact, and the upstream sources for the data
  behind it. Several require attribution to the original publisher.
- Twelve of the 523 databases carry upstream terms stricter than share-alike —
  `join-yoochoose` (CC BY-NC-ND 4.0), `join-apple-podcasts` and
  `join-atp-tennis` (CC BY-NC-SA 4.0), `join-imdb-full`, `join-imdb-ijs`,
  `join-imdb-small`, `join-imsa`, `join-indycar`, `join-nascar-cup`,
  `join-nascar-trk` (non-commercial, several also no-redistribution), and
  `join-gbif-biodiversity`, `join-gbif-species` (mixed per occurrence record).
  Those terms bind the source data, not the derived vectors here, but if you
  intend to reconstruct or redistribute anything resembling the originals, start
  from the raw repository and read them.

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
