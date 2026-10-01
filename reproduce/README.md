# Reproducing the RT-J paper

This directory is the experiment code behind every figure and table of the
RT-J paper, with the cluster it was run on taken out. It is a cleaned subset of
the private research directory the paper was produced from: the scripts on the
path to a published number, and nothing else.

It is research code. It has no stability promise and it is not part of the `rt`
public API — the library is [`src/rt`](../src/rt) and the supported entry points
are in [`examples/`](../examples). What is here is the *protocol*: which context
configuration each task won, how many ensemble members, which baselines were
featurized how, which seeds.

## What produces what

| paper artifact | stage |
|---|---|
| intro figure, main baselines figure, their per-task appendix | [`scaling/`](scaling) — `fulltest` arms |
| extended-context baselines appendix | [`scaling/`](scaling) — `subsampled` arms |
| retriever and schema-semantics ablations | [`scaling/`](scaling) — `abl` arms |
| test-time-compute ensembling figure + per-task appendix | [`enscurve/`](enscurve) |
| default-vs-tuned context appendix table | [`valtest/`](valtest) |
| tuned+ensembled per-task table, RelBench leaderboard submission | [`leaderboard/`](leaderboard) |
| per-task context grid feeding the three above | [`tune/`](tune) |
| baseline features and the FAISS retrieval indices | [`baselines/`](baselines) |
| pretraining-ablation figures (masking rate, task mix) | **not here** — see "What is not reproducible" |

## Two expensive stages you can skip

Both are committed, so you do not have to re-run them:

- **[`tune/tuned_configs.json`](tune/tuned_configs.json)** — the 120-point
  context grid's result for each of the 21 tasks: the winning
  `(ctx, local_ctx_size, bfs_width, prefer_latest)`, the top four
  configurations, and the full validation score table. The grid behind it is
  roughly 200 GPU-hours.
- **[`valtest/results.json`](valtest/results.json)** — the default-vs-tuned
  appendix table.

Every later stage reads `tuned_configs.json` from this directory, so
`enscurve` (tuned variant), `valtest` and `leaderboard` all run without the
grid.

## The reduced per-figure series

[`series/`](series) holds the reduced data behind each figure and table, as
committed JSON: one file per arm, with the aggregate metric at every context
size (or ensemble size) and the per-task value underneath it. These were
produced by the `reduce.py` / `collect.py` scripts here from the paper's runs,
and they are what makes the numbers checkable without re-running anything.

The paper's own plotting scripts are not in this repository; they read our
private Weights & Biases projects, so they would be a dead pointer. `series/`
is the same data those plots were drawn from.

## What you need

| | |
|---|---|
| **Hardware** | one GPU per job. The paper's runs used 80 GB A100s and H100s; a smaller card needs a smaller `tokens_per_gpu` (the plans pass `2**18`), which costs time rather than changing a result. The LightGBM arms are CPU-only by construction (`plan.METHOD_DEVICE`); CPU is not an option for the RT jobs — see "Running a stage" below. |
| **Disk, downloaded** | ~51 GB preprocessed RelBench data for RT-J (`RT_PRE_DIR`; the `legacy/` subdirectory is another ~57 GB and is needed only for the two legacy checkpoints), ~19 GB raw RelBench data (`RT_RAW_DIR`, baseline featurizers only), 164 MB for the RT-J checkpoint. |
| **Disk, derived** | `RT_SHARE` grows to roughly 240 GB once every stage's inputs exist, dominated by the FAISS retrieval indices (~141 GB) and the baseline feature tables (~51 GB). Per-task result JSONs in `RT_OUT_ROOT` are kilobytes. |
| **Time** | the committed `series/`, `tune/tuned_configs.json` and `valtest/results.json` cost nothing. Re-running a stage's evaluations costs hundreds of GPU-hours; the 120-point context grid in `tune/` is roughly 200 GPU-hours on its own; re-pretraining is thousands. A single job is minutes to a few hours — one task of the subsampled RT arm, six context sizes up to 8192 over 8192 subsampled rows, took 2 min 23 s on one A100. |
| **Software** | `pixi install` at the repository root, and nothing else. No `module load`, no cluster, no Weights & Biases account; `reproduce/` imports only `rt` and public packages. |

## Configuration

Every path comes from an environment variable and nothing is defaulted — an
unset variable fails loudly rather than guessing
([`config.py`](config.py)):

| variable | what it points at |
|---|---|
| `RT_CKPT` | a local directory holding the RT-J `config.json` + `model.safetensors` (mirror [`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j)) |
| `RT_PRE_DIR` | preprocessed RelBench data ([`stanford-star/relbench-preprocessed`](https://huggingface.co/datasets/stanford-star/relbench-preprocessed)) |
| `RT_RAW_DIR` | raw RelBench data ([`stanford-star/relbench`](https://huggingface.co/datasets/stanford-star/relbench)), baseline featurizers only |
| `RT_SHARE` | a writable directory for derived inputs: baseline features, FAISS indices, TabICL checkpoints, the semantics-ablated data copy |
| `RT_OUT_ROOT` | a writable directory for per-task result JSONs |

[`docs/downloads.md`](../docs/downloads.md) is how to fetch the data and the
checkpoint. All of it is public.

## Running a stage

A stage is a `plan.py` that enumerates its jobs and a `reduce.py` /
`collect.py` that turns the per-task JSONs into a series under `series/`.
Running a plan as a module executes its jobs **sequentially, in this process**:

```bash
python -m reproduce.enscurve.plan     # 42 jobs, one GPU, sequential
python -m reproduce.enscurve.reduce   # -> series/enscurve/{default,tuned}.json
```

To see a stage's job list without running anything:

```bash
python -c "from reproduce.enscurve.plan import jobs; from reproduce.launch import describe; describe(jobs())"
```

`reproduce.scaling.plan.jobs` is the one that takes an argument — the arm names
to enumerate, `sorted(reproduce.scaling.plan.arms())` for all 24. A plan lists
only the jobs whose output is missing, so it prints `0 jobs` once a stage is
finished.

That is the honest default, not a scheduler. The paper's runs were submitted in
parallel to a Slurm cluster, and the partitions, QoS names, accounts, node
exclusions and per-database memory and wall-clock limits that made that work
are not in this directory: they described one cluster on particular nights and
would be worse than useless anywhere else.

To run a stage on your own scheduler, implement one function. A plan's `jobs()`
returns [`launch.Job`](launch.py) records — `name`, `target`
(`"module:function"`), `args` (every argument spelled out) — and nothing else:

```python
from reproduce.enscurve.plan import jobs
from reproduce.launch import Job, run

def my_launcher(job: Job) -> None:
    ...  # sbatch / k8s / ssh: run `python -c "from <mod> import <f>; <f>(**args)"`

run(jobs(), my_launcher)
```

Every job is idempotent per output file: a plan skips a task whose JSON already
exists, so re-running fills gaps only. `enscurve` and `leaderboard` jobs also
resume per ensemble seed from a `.state.npz` beside the output, and a `tune`
job resumes per grid entry.

Jobs are sized for one GPU each. The RT jobs need a GPU: `flex_attention` has
no fused bfloat16 CPU kernel, and `scaling/run.py` on CPU did not finish even
the smallest possible call — one task, 5 rows, a 128-cell context — inside 50
minutes when this subset was assembled. Treat CPU as unusable here rather than
slow. The LightGBM arms are CPU-only by construction
(`plan.METHOD_DEVICE`).

The reduce and collect layer, by contrast, needs no GPU and runs in seconds to
minutes: it was used to produce every file in `series/`, and
`tune/collect.py` and `valtest/collect.py` reproduce the two committed
artifacts byte-for-byte from the paper's grids.

## Dependency order

```
baselines (features, FAISS indices, TabICL checkpoints)
  └─ scaling    (baseline and vdb arms; the RT arms need only the checkpoint)
tune
  ├─ enscurve (tuned variant)   ─┬─ valtest
  ├─ enscurve (default variant) ─┘
  └─ leaderboard
```

`scaling`'s `abl/nosem` arm needs the semantics-ablated data copy, which
`scaling/plan.py` builds as its first job.

## Protocol constants

These are the part that matters scientifically, and they are all literals in
the code rather than values read off our filesystem:

- Context grid: `ctx ∈ {512…8192} × lcs ∈ {256…8192 | lcs ≤ ctx} × bw ∈ {8,32,128} × pl ∈ {T,F}` = **120 configurations per task**, scored on 4096 validation rows averaged over 4 context seeds ([`tune/plan.py`](tune/plan.py)).
- Shared default context: `(8192, 256, 32, prefer_latest=True)` ([`enscurve/plan.py`](enscurve/plan.py)).
- Scaling context sizes: RT 256–8192, baselines extended to 131072 cells ([`scaling/plan.py`](scaling/plan.py)).
- Ensembling: 16 context seeds off base seed 0 over a fixed 8192-row test subsample.
- Leaderboard: top-4 configurations × 4 context seeds on the full test split, predictions averaged per row.
- Everywhere: `shuffle_seed=0`, `context_seed=0`, `db_cutoff=None` (per-row temporal masking is the only trim), `num_walks=10_000`, `walk_length=20`, the 21 RelBench forecasting tasks of `pre_dir/db-task-lists/forecast.json`.

Metrics are computed on the sampler's normalized target scale, which for
regression equals RelBench's NMAE and for classification is AUROC. Only
`leaderboard/reduce.py` writes prediction CSVs and scores them with RelBench's
own evaluator.

## What agreement to expect

**Not bit-exactness.** Two changes after the paper's runs mean a fresh run
cannot be expected to match a published digit-for-digit:

- **The context sampler's random stream changed.** The BFS child sampler used
  to draw indices with replacement until it filled its quota; it now draws
  without replacement. The distribution of contexts is unchanged — for rows
  with fewer than `bfs_width` visible children the old loop already returned
  essentially all of them — but which rows land in a context at a given seed is
  not, so contexts and every feature computed from them differ.
- **Input normalization changed.** Numeric and datetime cells are z-scored with
  per-column statistics. Those statistics used to be computed over all rows of
  every table, which let the validation and test periods set the scale that
  training inputs were normalized with; they are now restricted to the train
  period. The published preprocessed datasets still carry the old
  normalization, so a reader who regenerates the data from raw is evaluating on
  slightly different inputs than the paper did.

**How much the normalization change is worth: a few tenths of a point.** We
measured it directly — the released `stanford-star/rt-j` checkpoint at the
released default context (`ctx=8192`, `lcs=256`, `bfs_width=32`,
`prefer_latest=True`), one context seed, run twice with nothing but the
preprocessed directory differing:

| task | metric | old | new | new − old | rows |
|---|---|--:|--:|--:|--:|
| rel-f1/driver-dnf | AUROC ↑ | 82.843 | 82.537 | −0.306 | 702 |
| rel-f1/driver-top3 | AUROC ↑ | 90.537 | 90.683 | +0.146 | 726 |
| rel-event/user-repeat | AUROC ↑ | 79.101 | 79.088 | −0.013 | 246 |
| rel-hm/user-churn | AUROC ↑ | 61.955 | 61.947 | −0.008 | 4096 |
| rel-f1/driver-position | nMAE ↓ | 38.667 | 39.107 | +0.440 | 760 |
| rel-trial/study-adverse | nMAE ↓ | 16.413 | 16.375 | −0.038 | 3098 |
| rel-avito/ad-ctr | nMAE ↓ | 42.624 | 42.117 | −0.507 | 1816 |
| **mean, 4 classification tasks** | AUROC ↑ | 78.609 | 78.564 | **−0.045** | |
| **mean, 3 regression tasks** | nMAE ↓ | 32.568 | 32.533 | **−0.035** | |

Every move is under 0.51 points, five of the seven are under 0.31, and the sign
is mixed — it is noise-scale, not a systematic correction. The caveat matters as
much as the number: this is one context seed over at most 4096 rows per task,
far noisier than the paper's full-test four-seed ensembles, so it bounds the
magnitude of the effect and does not correct any published figure. It also does
not cover the two legacy checkpoints (`rt-v1`, `rt-plurel`), whose data lives in
the `legacy/` tree and changes under the same fix.

**So what a reader should expect:** running this code against the published
preprocessed data, or against data regenerated from raw, lands within a few
tenths of a point of the published numbers. A difference of that size is the
expected outcome and not a sign that something is wrong; a difference of a
point or more is.

## What has actually been run

Being specific about this, because "it should work" is not a claim worth
printing:

- **Verified on a GPU.** One job of `scaling/run.py` — the subsampled RT arm on
  rel-f1/driver-dnf, six context sizes from 256 to 8192 — ran to completion in
  2 min 23 s on one A100 80 GB and wrote its result JSON. Against the committed
  series for that arm and task it agreed to within 0.003 AUROC at five of the
  six context sizes (0.017 at the smallest, 256), which is the sampler's seed
  noise and not a code difference.
- **Verified without a GPU.** `scaling/reduce.py` regenerates all 18 committed
  series files byte-for-byte from the paper's per-task JSONs and
  `enscurve/reduce.py` does the same for both ensemble curves — re-run from a
  clean checkout, with nothing in `series/` changing. `tune/collect.py` and
  `valtest/collect.py` reproduce the two committed intermediates byte-for-byte,
  and `leaderboard/reduce.py` scores all 21 tasks
  through RelBench's own evaluator. Every `plan.py` enumerates its job list
  (408 scaling jobs over 24 arms, 84 leaderboard, 42 enscurve, 21 tune, 35
  baselines when none of their outputs exist), and an unset environment
  variable fails loudly.
- **Untested.** The other stages' job bodies have not been run from this
  directory on a GPU: `enscurve/run.py`, the `tune` grid, the `leaderboard`
  jobs, and the baseline featurizers under `baselines/`. They are the paper's
  code, carried across with the cluster submission layer removed and `device`
  taken as an argument instead of hardcoded, and the reduce layer that consumes
  their output is verified — but we say untested because untested is what they
  are.

## What is not reproducible, and why

- **The pretraining-ablation figures.** They compare checkpoints from five
  multi-day multi-node pretraining runs that are not released; only the final
  RT-J checkpoint is. The arms differed from the base run in exactly one knob —
  multi-cell masking rate (0, 0.25, 0.75 against the base) and pretraining task
  mix (forecast-only, autocomplete-only against the base's full mix), each with
  10k-step early-stop patience, `lr=5e-4`, `swa_momentum=0.9995`,
  from scratch, 2**15 + 1 steps — and the curves read are `swa/nmae/val/mean`
  and `swa/auroc/val/mean` against step. The recipe is
  [`docs/train.md`](../docs/train.md) and the corpus ("the Join") is public, so
  the runs are repeatable in principle at a cost of thousands of GPU-hours. The
  submission code for them is not in this subset: it depended on a held
  multi-node allocation on a specific cluster, and a portable version of it
  would be fiction.
- **Six appendix figures** are schematics with no generating script.
- **Non-transferable intermediates.** A FAISS on-disk IVF index bakes the
  absolute path of its `.ivfdata` file at build time, and the semantics-ablated
  data is a tree of symlinks into `RT_PRE_DIR`; neither can be copied, both are
  rebuilt by scripts here (`baselines/build_vector_db.py`,
  `scaling/make_nosem_data.py`).
- **Databases whose licenses do not permit redistribution** are absent from the
  public release of the pretraining corpus.

## If you only want to check one number

Evaluate `stanford-star/rt-j` on the task in question at the context
configuration [`tune/tuned_configs.json`](tune/tuned_configs.json) lists for
it. [`examples/eval.py`](../examples/eval.py) is that run with the arguments
spelled out, and the quickstart in the top-level
[`README.md`](../README.md) is the smallest version of it.

## What was dropped from the research directory

Everything not on the path to a published number: an adapter/fine-tuning side
investigation and a second-generation `rel2tab` baseline library (together
about 70% of the original directory, referenced by no README and behind no
figure), the probe scripts that settled a featurizer question now written up in
[`baselines/README.md`](baselines/README.md), the featurizers for baselines that
did not make the paper (GNN text encoder, entity-only, PluRel, an LLM-agent
program, RT-TAPS), a BFS-access sampling probe, the `roach` Slurm submitter and
every `Resources(...)` literal, our scratch paths, hostnames, and the `rtv2`
Weights & Biases entity.
