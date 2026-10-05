# Baseline featurization

Everything the paper's tabular baselines and retriever ablations need before
any eval can run: the [`rt.rel2tab`](../../src/rt/rel2tab) library (label-matched featurizer + predictor
pairs that the evaluator drives exactly like an RT network), precomputed
per-row features for every RelBench task table, and the FAISS indices the
vector-similarity retriever arms read. The evals themselves are
[`../scaling`](../scaling).

```bash
# 0. one-time fetches (needs internet)
python -m reproduce.baselines.fetch_tabicl
RELBENCH_CACHE_DIR="$RT_SHARE/relbench-cache" \
    python -m reproduce.baselines.populate_relbench_cache

# 1. prove classic-RelBench row order matches the preprocessed data
RELBENCH_CACHE_DIR="$RT_SHARE/relbench-cache" \
    python -m reproduce.baselines.check_alignment

# 2. featurize, then build the indices
python -m reproduce.baselines.plan
```

Feature blobs land under `$RT_SHARE/features/<db>/{sql,rdblearn,rt}_features/`
and FAISS indices under `$RT_SHARE/vector_db/{rdblearn,rt}/`, one
`<table>_vectors.bin` + `<table>_meta.json` (+ `.index`) per task table. A
finished table is skipped at plan time, so re-running fills gaps only.

## What each featurizer is

- **`featurize_sql.py`** — the "LLM data scientist" features: the committed
  DuckDB queries in [`sql_queries/`](sql_queries), one module per database
  (written once by a coding agent against each schema), run over the full
  database and z-scored globally. This is the most reusable artifact here and
  depends on nothing in this repository but DuckDB.
- **`featurize_rdblearn.py`** — RDBLearn's depth-2 DFS features (aggregation
  primitives max/min/mean/count/mode/std, per-row temporal cutoffs from the
  task's time column), transformed by the fitted RDBLearn preprocessor, **minus**
  the raw entity key, the cutoff time, and the cutoff's calendar expansions
  (`<cutoff>.year/.month/.day/.dayofweek`), and z-scored over all rows in
  float64.

  Those cuts matter and are a finding, not a detail. RDBLearn's preprocessor
  output is meant for its own tree estimator. Fed to an in-context predictor,
  the int64-nanosecond time columns (order 1e18) blow up TabICL's float32
  per-context standardization, and the calendar columns let it extrapolate
  heavy-tailed targets in time, since every context row precedes the query. With
  them in, RDBLearn + TabICL's regression error *rose* with context size while
  LightGBM's on the same blobs did not. The symptom to watch for in any new
  featurizer is a TabICL arm that gets worse with more labels. The SQL features
  carry no calendar columns.
- **`featurize_rt.py`** — RT-J row embeddings (the masked target cell's
  final-layer state over a walk-free 256-cell local context), for the
  RT-similarity retriever arm.

Feature row `r` of a table is node `min_offset + r` in the preprocessed data;
`check_alignment.py` is what makes that mapping trustworthy for the two
classic-RelBench featurizers, and every featurize job asserts its row counts
against `table_info.json`.

## Why there are two `rel2tab`s

[`src/rt/rel2tab/`](../../src/rt/rel2tab) is the released, generalized baseline
library: more featurizers and predictors, configs, a registry. The
`rel2tab/` here is the narrow version the paper's numbers were actually produced
with — four `(featurizer, predictor)` pairs, no registry, and in particular the
LightGBM hyperparameters and the TabICL batching the published curves used. The
two have diverged in API and in defaults, so this copy stays: a reproduction
script has to run the code that produced the number, not its successor.

## Dependencies

The two classic-RelBench featurizers need classic `relbench` 2.x, `fastdfs`,
`duckdb` and `rdblearn`, whose pins conflict with the default environment's;
see [`pyproject.toml`](../../pyproject.toml)'s `baselines` extra and the
`featurize` pixi environment. Everything else runs in the default environment.

The FAISS indices are read by the sampler's opt-in `vecdb` cargo feature, which
the default build does not include:

```bash
maturin develop --release --features vecdb
```

A FAISS on-disk IVF index (built for every table over 50k rows) bakes the
absolute path of its `.ivfdata` file at build time, so the indices do not
survive a move of `$RT_SHARE`: after a move, delete `$RT_SHARE/vector_db` and
rebuild, or every `vdb_*` pass on a large table dies at index load.

## Cost

Measured on the paper's cluster: RT featurization 2h16 for rel-amazon and 2h40
for rel-hm on a b200, a minute for rel-f1; FAISS build 8 minutes (rdblearn) and
36 minutes (rt) on 16 CPUs; the SQL and RDBLearn featurizers are CPU-only and
memory-hungry (up to 400G for rel-event).
