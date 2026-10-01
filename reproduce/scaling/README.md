# Context-scaling evals

The eval family behind the paper's context-scaling figures: the intro figure,
the main baselines comparison and its per-task appendix, the extended-context
baselines appendix, and the retriever and schema-semantics ablations.

One arm = one curve. An arm is a method plus a context-sampler configuration,
over the same 21 RelBench forecasting tasks, at the paper's shared default
context (`local_ctx_size=256`, `bfs_width=32`, `prefer_latest=True`) unless the
arm ablates exactly that.

```bash
python -m reproduce.scaling.plan      # every arm, sequentially; see ../README.md
python -m reproduce.scaling.reduce    # -> ../series/scaling/<group>/<arm>.json
```

`plan.py`'s first job builds the semantics-ablated data copy; the `abl/nosem`
arm needs it. The baseline and `vdb_*` arms need
[`../baselines`](../baselines) to have run first — `plan.py` does not check,
and a baseline job without its feature blobs crashes at
`PrecomputedFeaturizer`.

Per-task results land at `$RT_OUT_ROOT/scaling/<arm>/<db>__<table>.json`: the
metric on the normalized scale and the mean number of in-context labels, per
context size.

## The arms

- **`fulltest/*`** (5 methods, ctx 256–8192, full official test splits) — the
  intro figure, the main baselines figure, the per-task appendix.
- **`fulltest_ext/<method>/<ctx>`** — the TabICL baseline arms past RT-J's
  training context (16k–131k cells) over the 9 regression tasks, one job per
  context size, merged into the `fulltest` series by `reduce.py`.
- **`subsampled/*`** (5 methods, fixed 8192-row test subsample, baselines to
  131072 cells) — the extended-baselines appendix. `subsampled/rt` doubles as
  the random-walk arm of the retriever ablation and the semantics-on arm, which
  is why `series/scaling/abl/{rw,sem}.json` reduce from the same directory.
- **`abl/*`** (RT only, 8192-row subsample):
  - `rand` — `num_walks=0`, `prefer_latest=False`: walk-free, uniformly random
    same-table fallback.
  - `bfs32` / `bfs256` — one BFS around the target (`local_ctx_size=8192`) at
    the stated width.
  - `vdb_rdblearn` / `vdb_rt` — FAISS-similarity seed selection over the
    corresponding feature space. Needs the sampler built with its opt-in
    `vecdb` cargo feature: `maturin develop --release --features vecdb`.
  - `nosem` — the derived data copy whose column-name embeddings are deranged
    (`make_nosem_data.py`).

Everything is `db_cutoff=None`, a single context seed (`context_seed=0`), no
tuning and no ensembling. Metrics are on the sampler's normalized target scale
(NMAE for regression, AUROC for classification), so nothing here writes
submission CSVs — that is [`../leaderboard`](../leaderboard).

## Cost

Measured on the paper's cluster, startup included, as an order of magnitude
rather than a promise. Full-test RT arm: 1h40–4h30 per large task on an
a100/b200 (rel-amazon, rel-stack, rel-hm), under 6 minutes for the tasks under
2000 test rows. Full-test TabICL arms: 1–5h per large task. Subsampled RT
passes and every RT ablation pass: 5–15 minutes. Subsampled TabICL at 131k
cells: up to 7h on the largest task. The extension pieces at 131k cells were
8–12h each on a b200. The whole stage is a few hundred GPU-hours, and
sequentially it is weeks — this is the stage worth putting on a scheduler.
