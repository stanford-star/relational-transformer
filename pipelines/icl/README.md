# In-context leaderboard pipeline

A pretrained RT checkpoint predicts a RelBench task with **no gradient step on
the target database**: the only per-task choice is how the context is sampled,
and that choice is made on validation. This pipeline is the protocol behind our
two in-context RelBench leaderboard entries, **RT-J (in-context)** and
**RT-PluRel (in-context)**, and it runs unchanged on your own data.

Three stages:

1. **Context search.** For each task, score all **120** context configurations
   — `ctx ∈ {512, 1024, 2048, 4096, 8192}` × `lcs ∈ {256, 512, 1024, 2048,
   4096, 8192 | lcs ≤ ctx}` × `bfs_width ∈ {8, 32, 128}` × `prefer_latest ∈
   {True, False}` — on the task's validation split: 4096 rows at
   `shuffle_seed=0` (the whole split where it is smaller), the prediction
   averaged over 4 context seeds, `db_cutoff=None`. AUROC ranks classification
   configurations, normalized MAE ranks regression.
2. **Test ensemble.** Run each task's **top-4** configurations with **4 context
   seeds** each (`member_context_seed(0, k)`, k = 0..3) on the **full official
   test split**, and average the 16 raw predictions per row.
3. **Package.** Sigmoid (classification) or denormalize (regression) the
   averaged prediction into a per-task CSV, score it with RelBench's own
   evaluator, and zip it with `relbench.submit`.

Stage 2 and 3 are the same code the RT-J paper's leaderboard ensemble runs:
this pipeline is a model-aware entry point over
[`reproduce/tune`](../../reproduce/tune) and
[`reproduce/leaderboard`](../../reproduce/leaderboard), which hold the one
implementation. [`models.py`](models.py) is the whole of what a model adds —
its checkpoint variable, the grid its configurations came from, and where its
outputs land.

## Reproduce ours

### What you need

| | |
|---|---|
| **Hardware** | one GPU per job for stages 1 and 2; stage 3 needs no GPU. A smaller card than an 80 GB A100 needs a smaller `tokens_per_gpu` than the `2**18` the plans pass, which costs time rather than changing a result. CPU is not an option for stages 1 and 2. |
| **Checkpoint** | the released flat [`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j) or [`stanford-star/rt-plurel`](https://huggingface.co/stanford-star/rt-plurel), mirrored into a local directory holding `config.json` + `model.safetensors`. |
| **Data** | preprocessed RelBench ([`stanford-star/relbench-preprocessed`](https://huggingface.co/datasets/stanford-star/relbench-preprocessed)), ~51 GB. |
| **Software** | `pixi install` at the repository root, and nothing else. |

Pin the dataset revision recorded in [`docs/downloads.md`](../../docs/downloads.md),
which also lists the checkpoint revisions each published number was produced
against.

Every path is an environment variable and nothing is defaulted — an unset
variable fails loudly:

| variable | what it points at |
|---|---|
| `RT_CKPT` | the local RT-J checkpoint directory |
| `RT_PLUREL_CKPT` | the local RT-PluRel checkpoint directory |
| `RT_PRE_DIR` | the preprocessed data directory |
| `RT_TASKS` | a JSON file of `[db, task]` pairs: the tasks to run |
| `RT_OUT_ROOT` | writable: per-unit result JSONs and resume state |
| `RT_SHARE` | writable: prediction CSVs and submission zips |

### The commands

```bash
export RT_CKPT=...                     # or RT_PLUREL_CKPT for the rt-plurel arm
export RT_PRE_DIR=... RT_OUT_ROOT=... RT_SHARE=...
export RT_TASKS="$(pixi run python -c \
  'from rt.data import get_mixture_path; print(get_mixture_path("relbench", "forecast"))')"

# stage 1 -- 21 jobs, one GPU each. Skip it: the result is committed.
pixi run python -m pipelines.icl.tune     rt-plurel
pixi run python -m pipelines.icl.collect  rt-plurel   # -> reproduce/tune/tuned_configs_rt-plurel.json

# stage 2 -- 84 units (21 tasks x 4 configs), one GPU each
pixi run python -m pipelines.icl.ensemble rt-plurel

# stage 3 -- average, score, write the CSVs, print the packaging command
pixi run python -m pipelines.icl.package  rt-plurel
pixi run python -m relbench.submit "$RT_SHARE/leaderboard-rt-plurel/preds" \
  --out "$RT_SHARE/leaderboard-rt-plurel/rt-plurel-icl.zip"
```

`rt-j` in place of `rt-plurel` runs the other arm, into `leaderboard/` instead
of `leaderboard-rt-plurel/`. `models.py` names the two; any other argument
fails with the list.

**Stage 1 is committed for both checkpoints, so you can start at stage 2.** The
120-point grid is roughly 200 GPU-hours per checkpoint, and its result is
[`reproduce/tune/tuned_configs_rt-j.json`](../../reproduce/tune/tuned_configs_rt-j.json)
and
[`reproduce/tune/tuned_configs_rt-plurel.json`](../../reproduce/tune/tuned_configs_rt-plurel.json):
per task, the winning configuration, the top four, and the full 120-entry
validation score table. The two files are not interchangeable — a context
configuration is only valid for the checkpoint it was tuned on — so each one
records the grid it came from and every stage asserts that the file it was
handed carries this model's grids.

### Where outputs land

| | |
|---|---|
| a stage-1 grid | `$RT_OUT_ROOT/no-wandb-entity/tune/<grid stem>/tuning.json` |
| a stage-2 unit | `$RT_OUT_ROOT/<leaderboard dir>/cfg<rank>/<db>__<table>.json` with its `.state.npz` |
| prediction CSVs | `$RT_SHARE/<leaderboard dir>/preds/<db>__<table>.csv` |
| submission zips | `$RT_SHARE/<leaderboard dir>/<model>-icl-{classification,regression}.zip` |
| per-task scores | [`results/<model>.json`](results) in this directory |

[`results/rt-plurel.json`](results/rt-plurel.json) is committed — the per-task
metric, the alignment guard, and the four configurations behind each task — so
the RT-PluRel column of the table below is checkable without running anything.
RT-J's equivalent record is
[`reproduce/series/leaderboard/top4x4.json`](../../reproduce/series/leaderboard/top4x4.json),
written by the same reducer from the paper's own units.

Both stages resume and every job is idempotent per output file: a plan lists
only the jobs whose output is missing, so re-running it fills gaps and prints
`0 jobs` once the stage is finished. A stage-1 job resumes per grid entry and a
stage-2 unit per context seed, so a lost job costs one validation pass or one
full-test pass at most.

### Expected scores

RelBench's own evaluator on the full official test splits, measured from the
packaged CSVs — 12 classification tasks and 9 regression tasks:

| entry | clf `roc_auc` (higher) | reg `nmae` (lower) |
|---|---:|---:|
| RT-J (in-context) | 0.744512 | 0.323543 |
| RT-PluRel (in-context) | 0.729828 | 0.329467 |

Context sampling is stochastic, so a fresh run of stage 2 will not match these
digit-for-digit. Pin the dataset revision and the checkpoint and expect
agreement to a few tenths of a point; a difference of a point or more is worth
reporting. The top-4 validation scores of a task span 0.0005–0.008 on most
tasks, so which four configurations make the ensemble is close to a tie-break
and moves with the grid.

### Measured cost

Both stages, measured on 80 GB A100s and B200s:

- **Stage 1**, per task: one sampler pass over 4096 validation rows, scored at
  every `ctx` up to 8192 off a prefix of the largest, is ~300 s on an A100 at
  `bfs_width=128`, ~200 s at 32 and ~140 s at 8 — almost all of it predict. A
  task with a full 4096 validation rows is ~10.5 h on an A100 (rel-amazon's
  item-ltv, item-churn and user-ltv came in at 10h29, 10h27 and 10h54) and ~4 h
  on a B200; smaller validation splits scale down with their rows
  (rel-event/user-repeat 1h10, rel-trial/study-outcome 1h40, the rel-f1 tasks
  ~15 min). Roughly 200 GPU-hours for the 21 tasks.
- **Stage 2**, per seed pass — a unit is four, and there are 84 units: the
  352k-row rel-amazon user tasks at ctx 8192 are ~2.3 h on an A100 and ~1.05 h
  on a B200, ~25 min at ctx 4096 on a B200; rel-stack/user-badge (255k rows,
  ctx 8192) ~1.6–2 h on an A100 and ~50 min on a B200;
  rel-amazon/item-churn (167k, 8192) ~70 min on an A100; rel-stack/post-votes
  (161k, ctx 1024–8192 at width 8) 10–75 min on an A100; rel-hm/item-sales
  (106k, ctx 2048–4096) 15–30 min on a B200; everything else minutes. The BFS
  width matters as much as the context size: width-8 configurations run 2–3x
  faster than width-128 ones at the same `ctx`.
- **Stage 3** is minutes on a CPU for all 21 tasks.

### No scheduler

Running a stage executes its jobs sequentially, in this process. That is the
honest default: our own runs were submitted in parallel to a cluster, and the
queue names and limits that made that work described one cluster on particular
nights.

To run a stage on your own scheduler, implement one function. A stage's job
list is [`reproduce.launch.Job`](../../reproduce/launch.py) records — `name`,
`target` (`"module:function"`), `args` (every argument spelled out) — and
nothing else:

```python
from pipelines.icl.models import MODELS
from pipelines.icl.tasks import task_list
from reproduce import config
from reproduce.launch import Job, run
from reproduce.leaderboard.plan import jobs

def my_launcher(job: Job) -> None:
    ...  # run `python -c "from <mod> import <f>; <f>(**args)"` however you like

model = MODELS["rt-plurel"]
run(
    jobs(
        ckpt=config.env(model.ckpt_env),
        configs_path=model.configs_path,
        grid_stem=model.grid_stem,
        task_list=task_list(),
        out_subdir=model.out_subdir,
    ),
    my_launcher,
)
```

### What has actually been run

Being specific, because "it should work" is not a claim worth printing:

- **Verified without a GPU.** `pipelines.icl.collect` regenerates both
  committed `tuned_configs_*.json` byte-for-byte from the 42 grids they were
  ranked from. `pipelines.icl.package rt-plurel` reduces all 84 RT-PluRel units
  end to end and writes 21 prediction CSVs that are byte-identical to the ones
  in our submitted package, scoring 0.729828 / 0.329467; `relbench.submit`
  validates them 12/12 and 9/9 and writes both zips. An independent
  `relbench.submit` score of the RT-J package's CSVs gives 0.744512 / 0.323543.
  Both stages enumerate their job lists (21 and 84 per model), a mismatched
  configurations file is refused, and an unset environment variable fails
  loudly.
- **Untested.** The two GPU stage bodies have not been run from this directory:
  the stage-1 grid (`rt.eval:main`) and the stage-2 unit
  ([`reproduce/enscurve/run.py`](../../reproduce/enscurve/run.py)). They are the
  code our own units were produced by, with the cluster submission layer removed
  and `device` taken as an argument, and the stage-3 layer that consumes their
  output is verified against those units — but untested from here is what they
  are.

## Run it on your own data

The protocol does not change. Three things do.

1. **Point `RT_PRE_DIR` at your own preprocessed directory.** The pipeline
   never fetches anything at run time: a `pre_dir` is a local directory of
   `rustler` artifacts that every job reads. To get one, run your databases
   through the preprocessing pipeline at
   [`pipelines/preprocess`](../preprocess) — that pipeline is being written as
   this one is, so the path may not exist in your checkout yet;
   [`docs/preprocess.md`](../../docs/preprocess.md) is the same step as a
   library call.

   The input it expects is a dataset in **RelBench v3 format**, the format
   `stanford-star/relbench-v1` and the Join ship in: parquet table files
   carrying only native dtypes, plus a manifest that is the sole source of
   relational metadata — the tables, their primary and foreign keys and time
   columns, and the task declarations (target column, task type, splits). That
   is all RT needs: there is no feature engineering step and nothing to
   describe twice. Temporal correctness comes from those time columns — a
   context is built with per-row temporal masking and `db_cutoff=None`, so a
   row never sees its own future.

2. **Point `RT_TASKS` at your own task list** — a JSON file of `[db, task]`
   pairs, the tasks to search and ensemble. Nothing else in the pipeline
   enumerates tasks.

3. **Run stage 1.** The committed configurations are ours, tuned on our
   checkpoints against RelBench's validation splits; they are not valid for
   your data, and stage 2 will not silently reuse them — `collect` writes the
   configurations file the model's entry in `models.py` names, and stage 2
   asserts the grids in it are that model's. Budget stage 1 accordingly: it is
   the expensive stage, and 120 configurations per task is the whole of it.

A new checkpoint, rather than a new dataset, is a fourth: add a `Model` to
[`models.py`](models.py) — its checkpoint environment variable, a grid stem
that no other model shares, the configurations file `collect` should write, and
the output directory and zip name its arm uses.

Two caveats worth knowing before you start.

- **Stage 3's scoring and packaging are specific to the official RelBench
  board.** `relbench.submit` validates a submission against RelBench's own
  task tables, so it has nothing to say about a task of yours. On your own data
  the per-task metric that stage 2 prints and the averaged predictions are the
  result; the CSV and zip steps are for the board.
- **The checkpoints are pretrained, not universal.** RT-J is pretrained on "the
  Join" and RT-PluRel on the synthetic PluRel corpus; neither corpus holds a
  RelBench database, which is what makes these entries in-context. A database
  unlike anything in pretraining is a fair test of that, and a disappointing
  in-context number on one is a result rather than a misconfiguration. [`docs/train.md`](../../docs/train.md)
  is the recipe if you would rather pretrain your own.
