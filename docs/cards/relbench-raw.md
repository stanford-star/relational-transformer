---
# The `license` field below covers THIS REPOSITORY'S PACKAGING ONLY (the
# selection, naming and archiving of the mirrored files). Every archive retains
# the licence and terms of its upstream source; nothing here relicenses
# third-party data. See the "Licence" section, which says so in prose.
license: cc-by-4.0
license_details: >-
  CC BY 4.0 applies to the packaging of this mirror only. Each archive retains
  the licence and terms of its upstream source (RelBench's sources and the TGB
  datasets each have their own). Check the upstream terms before redistributing
  any individual archive.
pretty_name: RelBench raw sources (mirror)
tags:
  - relational-deep-learning
  - relational-databases
  - relbench
  - raw-data
  - mirror
size_categories:
  - 10B<n<100B
# No `configs:` block: this repository holds .zip archives, not tables.
---

# RelBench raw sources (`relbench-raw`)

A mirror of the **upstream raw sources** behind RelBench databases and the TGB
datasets, kept so RelBench-format databases can be rebuilt from scratch without
depending on third-party hosts staying up. These are the inputs to the database
*construction* step, not to RT.

- Code: <https://github.com/stanford-star/relational-transformer>
- RelBench: <https://relbench.stanford.edu>
- TGB: <https://tgb.complexdatalab.com>

## What it is

| | |
|---|---|
| Files | 23 (22 zip archives + `.gitattributes`) |
| Size | ~30.6 GiB |

```
rel-arxiv/db.zip                       114.0 MB
rel-avito/rel-avito-raw-100k.zip       518.7 MB
rel-f1/relbench-f1-raw.zip               6.2 MB
rel-ratebeer/db.zip                      1.9 GB
rel-salt/db.zip                         35.3 MB
rel-stack/relbench-forum-raw.zip       700.4 MB
rel-trial/relbench-trial.zip             1.1 GB
tgb/tgbl-coin/db.zip                     1.9 GB
tgb/tgbl-comment/db.zip                  3.7 GB
tgb/tgbl-flight/db.zip                   2.6 GB
tgb/tgbl-review/db.zip                   1.3 GB
tgb/tgbl-review-v2/db.zip                1.3 GB
tgb/tgbl-wiki/db.zip                   383.5 MB
tgb/tgbl-wiki-v2/db.zip                384.4 MB
tgb/tgbn-genre/db.zip                  133.8 MB
tgb/tgbn-reddit/db.zip                 304.5 MB
tgb/tgbn-token/db.zip                  308.2 MB
tgb/tgbn-trade/db.zip                    6.5 MB
tgb/thgl-forum/db.zip                    6.5 GB
tgb/thgl-github/db.zip                   1.6 GB
tgb/thgl-myket/db.zip                    4.4 GB
tgb/thgl-software/db.zip                 3.7 GB
```

## Relation to the other repositories

Three layers, in order:

| layer | repository | what it holds |
|---|---|---|
| raw upstream | **this repository** | the original vendor dumps, as zips |
| RelBench format | [`stanford-star/relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1), [`stanford-star/tgb`](https://huggingface.co/datasets/stanford-star/tgb) | parquet tables + `manifest.yaml` |
| RT format | [`stanford-star/relbench-preprocessed`](https://huggingface.co/datasets/stanford-star/relbench-preprocessed) | `rustler` artifacts + text embeddings |

You want this repository **only** if you are rebuilding a RelBench-format
database from its original source. To train or evaluate RT, go straight to the
preprocessed repository; to re-run RT's preprocessing, use the RelBench-format
repository.

## How it was produced

Each archive is a byte mirror of an upstream download, uploaded with
`huggingface_hub` and committed one dataset at a time (the commit titles record
the source, e.g. *"Mirror the rel-ratebeer raw source"*). Nothing in this
repository is produced by this project's code, so there is no preprocessing
commit for it — and conversely, it is unaffected by the `rustler` normalization
fix (PR #4, commit `8030aa8`) that changes the preprocessed repositories.

## Revisions

The RT-J paper's results do not read this repository directly, but the
RelBench-format databases they do read were built from revision
**`f1d7228af23a22b9ece756fe5dda4dda79b711dc`** (2026-08-25, the head at the
time). Pin it:

```bash
pixi run hf download stanford-star/relbench-raw --repo-type dataset \
  --revision f1d7228af23a22b9ece756fe5dda4dda79b711dc
```

## Licence

> **Read this before redistributing anything here.** The `cc-by-4.0` in the
> front matter covers **this repository's packaging only** — the selection,
> naming and archiving of the mirrored files. It is *not* a licence for the
> data inside the archives.

**Every archive retains the licence and terms of its upstream source.** This is
a byte mirror of third-party dumps: nothing in it is this project's work, and
this repository does not and cannot relicense it. RelBench's underlying sources
and the TGB datasets each carry their own terms, some of them share-alike or
otherwise restrictive, and several require attribution to the original
publisher rather than to us.

So:

- To **use** an archive, follow its upstream licence.
- To **redistribute** an archive, check its upstream licence first. The
  packaging licence above does not grant you that right.
- Per-source terms for the RelBench-format databases are catalogued in
  `STATS/databases.parquet` on
  [`stanford-star/relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1);
  TGB's are documented at <https://tgb.complexdatalab.com>.

## Citation

Cite the upstream dataset you used, plus RelBench or TGB as appropriate.
