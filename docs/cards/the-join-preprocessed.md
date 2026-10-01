---
# CC BY-SA 4.0 is inherited, not chosen: about half the source databases are
# share-alike upstream and this artifact reproduces their text verbatim. The
# "Licence" section below says so in prose.
license: cc-by-sa-4.0
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
| Files | 4,189 |
| Size | 278,794,986,854 bytes (~259.6 GiB) |
| Text embedder | `sentence-transformers/all-MiniLM-L12-v2` (384-d) |

Per-file-kind totals, which is what a partial download is planned against:

| file | total |
|---|---|
| `nodes.rkyv` | ~195.1 GiB |
| `text_emb_all-MiniLM-L12-v2.bin` | ~29.8 GiB |
| `p2f_adj.rkyv` | ~26.1 GiB |
| `offsets.rkyv` | ~5.3 GiB |
| `text.json` | ~3.2 GiB |

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
  text.json                            # the source strings (not needed to train)
```

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
has the `--include` patterns that skip `text.json` and the embedders you are not
using (~256 GiB instead of ~260), and the revision pins below.

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

### The temporal-normalization fix does not change this dataset

`rustler` PR #4 (`rustler: column stats from the train period only`, commit
`8030aa8`, merged 2026-10-01) is a temporal-leakage fix: before it, per-column
z-scoring statistics and the global datetime statistics were computed over the
validation and test rows as well as the train period. It changes
[`relbench-preprocessed`](https://huggingface.co/datasets/stanford-star/relbench-preprocessed)
and
[`plurel-preprocessed`](https://huggingface.co/datasets/stanford-star/plurel-preprocessed).
**It does not change this one, and this one was therefore not regenerated.**

The fix has two halves and neither engages here. It cuts database-table
statistics at the manifest's `val_timestamp`, and all 639 manifests in
`stanford-star/the-join` have `val_timestamp: null`, which the fix explicitly
leaves on full-table statistics. And it drops validation and test *task* tables
out of the statistics loop, but this collection has no validation or test
splits at all — the 6,023 forecasting tasks ship a `train.parquet` and nothing
else. So the pre-#4 and post-#4 preprocessors produce byte-identical artifacts
for every database here, and the published revision is the one the current code
reproduces.

A second merged change, PR #3 (`rustler: draw BFS children without rejection
sampling`, `fly.rs`), affects the *sampler*, not this data, but means contexts
drawn over it are not bit-identical to the paper's.

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

**CC BY-SA 4.0**, inherited from
[`stanford-star/the-join`](https://huggingface.co/datasets/stanford-star/the-join)
and, through it, from the upstream sources.

The share-alike is not a choice made here. Of the 523 databases published in
this repository, **264 (50.5%)** are copyleft, non-commercial or no-derivatives
upstream:

| upstream source | databases | declared licence |
|---|---:|---|
| Spider 1.0 | 143 | CC BY-SA 4.0 |
| BIRD | 58 | CC BY-SA 4.0 |
| Stack Exchange dumps | 40 | CC BY-SA 4.0 |
| Wikipedia-derived | 5 | CC BY-SA 4.0 |
| OpenStreetMap / Overture / GeoNuclearData | 4 | ODbL 1.0 |
| Lahman / Baseball Databank, other | 3 | CC BY-SA 3.0 |
| exploit-db (Offensive Security) | 1 | GPL 2.0 |
| non-commercial or no-derivatives (2 of them also share-alike) | 12 | see the warning below |
| unrestricted (CC0, CC BY, MIT, BSD, public domain) | 259 | — |

That is 254 share-alike and 12 non-commercial / no-derivatives, overlapping in
two, so 264 encumbered and 259 unrestricted.

Per-database attribution — the `license` and `source_url` columns for all 639
databases — lives on the raw repository,
[`stanford-star/the-join`](https://huggingface.co/datasets/stanford-star/the-join),
in `STATS/databases.parquet`. It applies to the derived artifacts here.

**Why the obligation carries into the preprocessed form.** This is a format
conversion, not an independent work. `text.json` is a verbatim, deduplicated
intern table of every string cell value in the source database, and
`nodes.rkyv` stores the index into it, so every string and free-text column is
exactly reconstructible — for the Spider, BIRD and Stack Exchange databases,
whose content is overwhelmingly text, the share-alike material is present
essentially in full, over a reproduced schema and foreign-key graph. Numeric
and datetime cells are z-scored and the statistics are not serialized, so
absolute scale is unrecoverable, and primary-key columns are dropped. Neither
makes the textual content derived.

What share-alike does and does not do here: it applies when you redistribute
this dataset or a modified version of it. It does **not** restrict commercial
use, and it does **not** reach model weights trained on the data — the
checkpoints released alongside it are CC BY 4.0.

> **Twelve databases carry terms stricter than share-alike, and no outbound
> licence cures them.** `join-yoochoose` is CC BY-NC-ND 4.0, and
> no-derivatives is in tension with preprocessing itself.
> `join-apple-podcasts` and `join-atp-tennis` are CC BY-NC-SA 4.0.
> `join-imdb-full`, `join-imdb-ijs`, `join-imdb-small` (IMDb),
> `join-imsa`, `join-indycar` (Racing-Reference) and `join-nascar-cup`,
> `join-nascar-trk` (NASCAR Digital Media) assert non-commercial use and, in
> several cases, **no redistribution**. `join-gbif-biodiversity` and
> `join-gbif-species` are mixed CC0 / CC BY / CC BY-NC per occurrence record.
> If your use is commercial, or you intend to redistribute, exclude these
> before you do.

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
