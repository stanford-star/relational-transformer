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
