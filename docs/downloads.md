# Downloads

Our HuggingFace org [`stanford-star`](https://huggingface.co/stanford-star)
provides raw data, preprocessed data, and model checkpoints.

**Data is downloaded up front, by you; only checkpoints are fetched on demand.**
A `pre_dir` is always a local directory. Preprocessed datasets run to hundreds
of GiB and every rank and dataloader worker of a run reads them, so on-demand
fetching meant thousands of Hub requests per run (HTTP 429 rate limits, even
when the bytes were already cached, because each call still revalidates over the
network) and a separate copy per machine. One explicit `hf download` into a path
you choose is faster, and the data a run used is a directory you can inspect.

Download the preprocessed data you need with the `hf` CLI:

```bash
# Preprocessed "the Join" (rustler artifacts, ready for RT) -- the pretraining data
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed

# Preprocessed RelBench (rustler artifacts, ready for RT) -- validation/eval data
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed
```

Those are the paths the scripts default to (`--train.pre-dir data/the-join-preprocessed`,
`--eval.pre-dir data/relbench-preprocessed`); pass your own to put them elsewhere.
Each repo also ships its curated task lists under `db-task-lists/`, so they
arrive with the data they refer to.

The full preprocessed Join is **~260 GiB** at the current revision, since the
2026-09-12 trim to the 523 databases under the 5 GB per-database cutoff (it was
~1.5 TiB before). To fetch only what a run needs, keep
the core rustler artifacts plus the one text embedder you train with, and skip
`text.json`:

```bash
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed --max-workers 16 \
  --include "db-task-lists/*" "*/meta.json" "*/table_info.json" "*/column_index.json" \
            "*/nodes.rkyv" "*/offsets.rkyv" "*/p2f_adj.rkyv" \
            "*/text_emb_all-MiniLM-L12-v2.bin"
```

Narrow it further with `--include "<db>/*"` per database (a `db-task-lists/*.json`
entry names the dbs a mixture needs). Sizes across the 523 databases:
`nodes.rkyv` ~195 GiB, `text_emb_all-MiniLM-L12-v2.bin` ~30 GiB, `p2f_adj.rkyv`
~26 GiB, `offsets.rkyv` ~5 GiB, `text.json` ~3 GiB.

Raw data (only needed to re-run preprocessing yourself, see
[preprocess.md](preprocess.md)) and checkpoints:

```bash
# Raw "the Join" (639 databases in RelBench format)
pixi run hf download stanford-star/the-join --repo-type dataset

# Raw RelBench databases (RelBench format)
pixi run hf download stanford-star/relbench-v1 --repo-type dataset

# The RT-J checkpoint
pixi run hf download stanford-star/rt-j --repo-type model
```

Checkpoints are the one thing still resolved from the Hub on demand: a single
small file fetched once by one process, so `load_rt_model("stanford-star/rt-j")`
and `--model.load-ckpt-path stanford-star/rt-j` keep working without a manual
download. The preprocessor also reads its *raw* inputs straight from the Hub.

## Revisions the published results were produced against

These repositories are rewritten in place as databases are added, dropped or
re-preprocessed, so an unpinned download is not necessarily the data a published
number came from. `the-join-preprocessed` was trimmed to the 523 databases under
the 5 GB per-database cutoff on 2026-09-12, dropping 116 of them; a reader who
takes the head today gets a different dataset than an earlier reader did.

Every number in the RT-J paper was produced against these revisions:

| repository | revision | date |
|---|---|---|
| `stanford-star/the-join-preprocessed` | `ba5574ba659f0ef8b592793ed2ac9cb78dc87100` | 2026-09-12 |
| `stanford-star/relbench-preprocessed` | `1016626ddb30c027b92458bf866903850cc205e1` | 2026-08-08 |
| `stanford-star/plurel-preprocessed` | `9d70172425b44053f1270b19094ba6dcf7d66464` | 2026-09-12 |
| `stanford-star/the-join` | `ec028dddca63c7eeb49bbce3d7a713783e165eea` | 2026-08-04 |
| `stanford-star/relbench-v1` | `d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600` | 2026-08-25 |
| `stanford-star/plurel` | `ae2f0b04f71ec17aebf6cf1fa8259fa655994b58` | 2026-09-08 |
| `stanford-star/relbench-raw` | `f1d7228af23a22b9ece756fe5dda4dda79b711dc` | 2026-08-25 |
| `stanford-star/rt-j` | `360798f5335975fdcae73e3ed58cf03366dbc082` | 2026-09-14 |
| `stanford-star/rt-plurel` | `d27c97b045fc4f504848f15c730acb87970aac1d` | 2026-09-11 |

Pass the revision to every download that has to be reproducible:

```bash
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --revision ba5574ba659f0ef8b592793ed2ac9cb78dc87100 \
  --local-dir data/the-join-preprocessed

pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --revision 1016626ddb30c027b92458bf866903850cc205e1 \
  --local-dir data/relbench-preprocessed
```

Two further things fix a result besides the data: the checkpoint, and the
`rustler` commit the data was preprocessed with and the contexts were sampled
with. The published preprocessed repositories above were all built **before**
`8030aa8` (`rustler: column stats from the train period only`), which restricted
z-scoring statistics to the train period. Regenerating at or after that commit
changes the data without changing the on-disk format — but not every repository
is affected:

| repository | changed by `8030aa8`? |
|---|---|
| `relbench-preprocessed` | **yes** — 34 of 50 database tables are time-indexed, and 27 of 40 numeric columns shift by more than 0.1 old-std (worst 1.43). `legacy/_transformed/` moves too, so RT-v1 results are pinned to this revision as well. |
| `plurel-preprocessed` | **yes** — PluRel manifests carry a real `val_timestamp`. |
| `the-join-preprocessed` | **no** — all 639 manifests have `val_timestamp: null` and the collection has no val/test splits, so neither half of the fix engages. The published revision is what the current code reproduces. |

The downstream effect, measured with the released `rt-j` at its released default
context, same commit and seed, only `pre_dir` differing: mean **−0.045 AUROC**
and **−0.035 nMAE** over seven paired tasks, no task moving more than 0.51, sign
mixed and largely cancelling. That is one context seed over at most 4,096 rows
per task, so it bounds the magnitude rather than correcting any published
number; the per-task table is on the
[`relbench-preprocessed` card](https://huggingface.co/datasets/stanford-star/relbench-preprocessed).

Without `--local-dir` these land in the shared HuggingFace cache
(`~/.cache/huggingface/hub`, or `$HF_HOME`), which is what you want for
checkpoints and raw preprocessing inputs. For a `pre_dir`, use `--local-dir` so
the path you pass to the scripts is a plain directory. Useful flags:
`--include`/`--exclude` (glob patterns to grab a subset), `--max-workers`
(parallel downloads), and `--revision` (pin a branch, tag, or commit).
