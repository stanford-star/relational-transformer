# Preprocessing a collection

Building a whole preprocessed collection — the three that RT-J's pretraining
reads — rather than one database. [`plan.py`](plan.py) enumerates the work as
two jobs per database and [`run.py`](run.py) is the two job bodies.

[`examples/preprocess.py`](../../examples/preprocess.py) is the minimal example:
one database, one call, end to end. It is the right starting point for your own
single database. This directory is what the released collections were built
with, at the scale they were built at: 639 raw databases for the Join, 2,000 for
PluRel.

The library reference for the format and for every argument is
[`docs/preprocess.md`](../../docs/preprocess.md).

## Reproduce ours

### Why two stages per database

A database is preprocessed by two jobs, not one, because they want different
hardware and the work is extremely lopsided:

| stage | hardware | what it does | scales with |
|---|---|---|---|
| `rustler` | 1 CPU, **no GPU**, memory ~6× the expected output | parquet + `manifest.yaml` → the on-disk tensor format, plus a `text.json` intern table | output size |
| `embed` | **1 GPU**, a few CPUs | `text.json` → `text_emb_<embedder>.bin`; the string cells in `nodes.rkyv` become indices into it | text volume |

Run as one job they get neither: the rustler stage would hold a GPU it never
touches, and a GPU-count ceiling then caps how many databases can run at once.
Split, the CPU stage runs as wide as your cores and the GPU stage is a short
queue behind it.

They rank differently too, which is why neither is a proxy for the other.
Slowest to rustle in the Join build were `join-se-electronics` (81 GiB of
output, 1743 s), `join-tpch` (72 GiB, 1332 s) and `join-se-english` (57 GiB,
1225 s); slowest to embed were `join-open-food-facts` (7173 s from only 10 GiB
of output), `join-bird-codebase-comments` (5983 s) and `join-unpaywall`
(4727 s).

### 1. Fetch the raw collection

Raw databases are read from a local directory — one subdirectory per database,
each with a `manifest.yaml`.

```bash
pixi run hf download stanford-star/the-join --repo-type dataset \
  --revision ec028dddca63c7eeb49bbce3d7a713783e165eea --local-dir data/the-join
pixi run hf download stanford-star/plurel --repo-type dataset \
  --revision ae2f0b04f71ec17aebf6cf1fa8259fa655994b58 --local-dir data/plurel
pixi run hf download stanford-star/relbench-v1 --repo-type dataset \
  --revision d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600 --local-dir data/relbench-v1
```

The raw download is the long pole and does not parallelise well. Pin the
revisions ([`docs/downloads.md`](../../docs/downloads.md)); these repositories
are rewritten in place as databases are added and dropped.

### 2. Run the plan

```bash
# the Join: every job still outstanding, sequentially, in this process
pixi run python -m pipelines.preprocess.plan

# the job list without running anything
pixi run python -c "from pipelines.preprocess.plan import the_join_jobs; from reproduce.launch import describe; describe(the_join_jobs())"
```

`plan.py` has one entry point per released collection — `the_join_jobs()`,
`plurel_jobs()`, `relbench_jobs()` — each a `jobs(...)` call with the repository,
output directory, embedder (`all-MiniLM-L12-v2`) and batch size (1024) spelled
out. Edit the `__main__` line to pick one, or call `jobs(...)` with your own
arguments.

Every job is idempotent and the plan is the resume: it lists only what is
outstanding, so a crash, an out-of-memory kill or an interrupted sweep all have
the same fix, which is to run it again.

- rustler is done when `text.json` exists — it is written last.
- a database is done when its `meta.json` names an embedding file that exists.
  Not when its directory exists: a run interrupted between the two stages leaves
  a directory that looks finished.

A finished database holds eight files:

```
<db>/  meta.json  table_info.json  column_index.json
       nodes.rkyv  offsets.rkyv  p2f_adj.rkyv
       text.json  text_emb_all-MiniLM-L12-v2.bin
```

### 3. Measured cost

Measured over the Join build, 639 databases. These are CPU/GPU seconds of real
work, not wall clock — at any real concurrency the sweep finishes far sooner
than the totals suggest.

| | rustler | embed |
|---|---|---|
| total | 3.8 h | 10.6 h |
| share of the work | 26% | 74% |
| median database | 2 s | 4 s |
| mean | 47 s | 128 s |
| slowest | 1743 s | 7173 s |
| top 10 databases | 64.5% of the stage | 87.9% of the stage |

So **~14.5 hours of single-stream work for the whole Join**, and essentially all
of the makespan lives in the top ~20 databases: roughly 590 of the 639 together
are a few percent of the work. Scheduling the tail cleverly buys nothing.

Disk, and the reason to check it before starting:

| collection | databases | raw | preprocessed |
|---|---|---|---|
| the Join | 639 | 28 GiB | ~1.4 TiB over all 639; ~256 GiB for the 523 the RT-J mixture names |
| PluRel | 2,000 | small — synthetic databases carry little text | ~40 GiB |
| RelBench | 7 | 10 GiB | ~230 GiB |

The published `the-join-preprocessed` carries the **523 databases under the 5 GB
per-database cutoff** — the same set `rt.data.get_mixture("the-join", "rt-j")`
names, 13,243 tasks. Preprocessing the full raw collection and then pointing a
run at that mixture reads only those 523; it costs disk, not correctness.

Embedding throughput, measured on 2M texts: ~929 texts/s on one Quadro RTX 8000
and ~849 on one 2080 Ti, reaching 3,769 and 3,643 on six cards of each — so six
GPUs are worth about four, and the card barely matters (9% between those two).
One GPU per database with many databases at once beats many GPUs per database.

### What the released mixtures are, and what is not rebuildable

`rt.data.get_mixture(collection, name)` reads the mixtures vendored in the
package; nothing has to be downloaded and nothing can drift from the data.
`all`, `forecast` and `autocomplete` are derived from the databases' `meta.json`
files, so this pipeline's output determines them. Two are curated and are **not**
derivable from the build:

- `the-join/rt-j` — the 13,243 pairs over the 523 databases phase 2 trains on.
- `plurel/rt-plurel-train` — 86,211 pairs over 1,900 of PluRel's 2,000
  databases. The filter that produced it is described on the
  [`plurel-preprocessed` card](../../docs/cards/plurel-preprocessed.md) and
  needs per-column statistics that the published collection does not carry.

Both are vendored, so you do not have to rebuild either.

### What has actually been run

Being specific, because "it should work" is not a claim worth printing:

- **Verified.** Both stages, end to end, on one real database: `rustler/rel-f1`
  then `embed/rel-f1` through `run_sequential`, on CPU, in 1 s + 10 s, writing
  all eight files. Re-planning afterwards reports that database as having no
  outstanding job, so the resume is the plan. `jobs()` enumerates the 7-database
  RelBench collection as 14 jobs. A missing `raw_dir`, a database with no
  `manifest.yaml`, and an embed stage with no rustler output to embed each fail
  with an assertion naming the fix.
- **Not verified here.** A full 639-database run, and the embed stage on a GPU
  (the verification above ran it on CPU, which is the same code path minus the
  device). The costs in the table are measurements from the released build, not
  from this plan.

## Run it on your own data

### What the input must look like

A raw database is a **RelBench-format directory** and nothing else is read:

```
<db>/
  manifest.yaml          # the sole source of relational metadata
  db/<table>.parquet     # one file per table, native dtypes only
  tasks/<task>/manifest.yaml
  tasks/<task>/<split>.parquet
```

`manifest.yaml` is where tables, primary keys, foreign keys, time columns and
semantic types live; the parquet files carry only native dtypes and are not
inspected for structure. A task manifest names its `entity_table`,
`target_col`, `task_type` (`binary_classification` or `regression` — anything
else is skipped), `time_col`, and `kind` (`autocomplete` for predicting a column
of a database table, otherwise a forecast task). If a target column is constant
and gets dropped in preprocessing, the task is reported and left out rather than
failing the build.

[`byod/`](../../byod) walks a database from DuckDB through tasks in SQL to a
forward pass, and is the better starting point if you are writing a
`manifest.yaml` for the first time.

### Pointing it at your own collection

```python
from pipelines.preprocess.plan import jobs
from reproduce.launch import run_sequential

run_sequential(
    jobs(
        raw_dir="data/my-collection",
        out_dir="data/my-collection-preprocessed",
        source_repo="my-org/my-collection",
        embedder="all-MiniLM-L12-v2",
        batch_size=1024,
    )
)
```

`raw_dir` is scanned for `*/manifest.yaml`, so a directory holding one database
is a one-database collection and works the same way. `source_repo` is recorded
in each `meta.json` as provenance and is not fetched from.

Two things to get right:

- **`embedder` is part of the data.** A checkpoint is tied to the embedder its
  training data was built with; `all-MiniLM-L12-v2` at `d_text` 384 is what
  every released RT checkpoint expects. A different embedder means re-running
  this pipeline from the raw collection — the published preprocessed
  repositories ship embeddings and no readable strings, so a downloaded tree
  cannot be re-embedded.
- **`batch_size`** is the text-embedding batch; lower it if the embed stage runs
  out of GPU memory. It does not affect the output.

To run the jobs anywhere other than this process, `jobs()` returns
[`launch.Job`](../../reproduce/launch.py) records — `name`, `target`
(`"module:function"`), `args` — and nothing else, so a launcher is one function:

```python
from reproduce.launch import Job, run

def my_launcher(job: Job) -> None:
    ...  # run `python -c "from <mod> import <f>; <f>(**args)"` however you like

run(jobs(...), my_launcher)
```

The rustler jobs want CPU and memory and no GPU; the embed jobs want one GPU
each. That split is the whole reason the plan emits them separately.

Then point a training or evaluation run at `out_dir` as its `pre_dir` — the
layout is identical to a downloaded collection, so your own output and a
published one are interchangeable.
