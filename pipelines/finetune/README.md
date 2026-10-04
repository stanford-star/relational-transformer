# Per-task fine-tuning

Fine-tune one Relational Transformer checkpoint per RelBench task, tune that
task's context against the fine-tuned weights, refit on `train + val`, and
score an 8-seed test ensemble through RelBench's own evaluator. This is the
pipeline behind our two fine-tuned RelBench leaderboard entries, RT-J and
RT-PluRel.

**These are leaderboard results, not paper claims.** The RT-J paper reports the
in-context model; nothing here changes a number in it.

Four stages per task, all of them existing `rt` entry points — `run.py` only
derives the values between them. `db_cutoff=None` throughout: per-row temporal
masking is the only trim of the database a context is built from.

1. **Selection arm** (`inner`) — `rt.train.main` on `train`, delta-fine-tuned
   from the warm start: Muon, constant lr 5e-4, wd 0.1, batch 256, an EMA of
   the weights (`swa_momentum=0.9999`), up to 50k steps. Every batch draws its
   context shape from `ctx {128, 256, 512, 1024} x local ctx
   {128, 256, 512, 1024} x bfs width {16, 64, 256} x prefer-latest {F, T}`.
   Every 100 steps the EMA net is scored on 1024 val rows under the two
   endpoint shapes `(1024, 1024, 256, False)` and `(128, 128, 16, True)`, each a
   4-seed context ensemble; the best step by the task's metric is kept as
   `best_swa_<task_type>.safetensors`, and the arm stops after 10k steps without
   an improvement in either shape.
2. **Context search** (`tune`) — `rt.eval.main` on `val` with that checkpoint
   frozen: the 60 shapes of the grid above (local ctx <= ctx), each a 4-seed
   ensemble over 4096 val rows at `shuffle_seed=1`, so not the rows the step was
   chosen on. The winner lands in `tuning.json`.
3. **Reporting arm** (`outer`) — `rt.train.main` on `train + val`, same recipe,
   no evaluation, for the selected step scaled by the row ratio
   `(train + val) / train`; the EMA net at the last step is the model.
4. **Test ensemble** (`test`) — `rt.eval.main` on the full `test` split under
   the chosen shape, 8 context seeds averaged before the sigmoid or
   denormalization, written as the leaderboard prediction table
   `<db>__<task>.csv`.

`inner/selection.json` records the selected step, the row counts, the refit
budget and the chosen context for each task.

**Every stage is idempotent, and that is what makes the pipeline usable on a
preemptible machine.** A rerun of a job resumes the stage it was in — `rt.train`
from `resume.pt`, the context search per grid entry, the test ensemble per seed
— and skips every stage whose output already exists. A job whose prediction
table exists is not planned at all, so `plan.py` prints `0 jobs` once a model is
finished and fills only gaps otherwise. To move or interrupt a run, kill it and
run it again; nothing is lost but the minutes since the last checkpoint.

## Reproduce ours

One GPU per job. Two environment variables, both unset by default and both
failing loudly if missing ([`config.py`](config.py)):

| variable | what it points at |
|---|---|
| `RT_PRE_DIR` | preprocessed RelBench data ([`stanford-star/relbench-preprocessed`](https://huggingface.co/datasets/stanford-star/relbench-preprocessed)) |
| `RT_OUT_ROOT` | a writable directory for checkpoints, tuning results and prediction tables |

The warm-start checkpoints are Hub ids resolved on demand, not paths:
`rt-j` warm-starts from
[`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j) and
`rt-plurel` from
[`stanford-star/rt-plurel`](https://huggingface.co/stanford-star/rt-plurel)
([`plan.py`](plan.py) `MODELS`). [`docs/downloads.md`](../../docs/downloads.md)
is how to fetch the data.

```bash
export RT_PRE_DIR=data/relbench-preprocessed
export RT_OUT_ROOT=out/finetune

# 21 jobs per model, one GPU each, run sequentially in this process
pixi run python -m pipelines.finetune.plan rt-j
pixi run python -m pipelines.finetune.plan rt-plurel

# gather the prediction tables, score them, write the submission zips
pixi run python -m pipelines.finetune.collect rt-j
pixi run python -m pipelines.finetune.collect rt-plurel
```

To see a model's job list without running anything:

```bash
pixi run python -c "from pipelines.finetune.plan import jobs, TASKS, PROJECT; \
from pipelines.finetune import config; from reproduce.launch import describe; \
describe(jobs(models=['rt-j'], tasks=TASKS, project=PROJECT, \
pre_dir=config.pre_dir(), out_root=config.out_root(), tokens_per_gpu=2**17, \
num_workers=8, eval_num_workers=2, selection_steps=50_000, \
patience_steps=10_000, eval_freq=100, eval_rows=1024, \
selection_ensemble_size=4, tune_rows=4096, test_ensemble_size=8, seed=0))"
```

That sequential loop is the honest default, not a scheduler. A plan's `jobs()`
returns [`reproduce.launch.Job`](../../reproduce/launch.py) records — `name`,
`target` (`"module:function"`), `args` with every argument spelled out — so to
run the 21 jobs in parallel on your own scheduler, implement one function and
hand it to `reproduce.launch.run`; per-stage idempotence means a job your
scheduler preempts and resubmits picks up where it stopped.

Where the outputs land, under `$RT_OUT_ROOT`:

| path | what |
|---|---|
| `no-wandb-entity/finetune/<model>-<db>-<task>-{inner,tune,outer,test}/` | one directory per stage: `resume.pt`, the selected and final checkpoints, `selection.json`, `tuning.json` |
| `no-wandb-entity/finetune/<model>-<db>-<task>-test/eval_out/<db>__<task>.csv` | that task's leaderboard prediction table |
| `leaderboard/finetune/<model>/` | the 21 prediction tables `collect.py` gathers |
| `leaderboard/finetune/<model>-{classification,regression}.zip` | the submission packages, written only once all 21 tables validate |

Attach those two zips to a RelBench
[submission issue](https://github.com/stanford-star/relbench/issues/new?template=submit.yml);
"In-context?" is **No** for both models, since each trains on the target
database.

### Expected scores

Means over the 12 classification tasks (test ROC-AUC) and the 9 regression
tasks (test NMAE, lower is better), as `relbench.submit`'s evaluator reports
them:

| model | clf roc_auc | reg nmae |
|---|---|---|
| RT-J (fine-tuned) | **0.7902** | **0.2711** |
| RT-PluRel (fine-tuned) | 0.7853 | 0.2757 |

Per task:

| task | metric | rt-j | rt-plurel |
|---|---|---|---|
| rel-amazon/item-churn | roc_auc | 0.8316 | 0.8319 |
| rel-amazon/user-churn | roc_auc | 0.7146 | 0.7134 |
| rel-avito/user-clicks | roc_auc | 0.6681 | 0.6589 |
| rel-avito/user-visits | roc_auc | 0.6689 | 0.6675 |
| rel-event/user-ignore | roc_auc | 0.8561 | 0.8085 |
| rel-event/user-repeat | roc_auc | 0.7984 | 0.8029 |
| rel-f1/driver-dnf | roc_auc | 0.8277 | 0.8209 |
| rel-f1/driver-top3 | roc_auc | 0.9171 | 0.9001 |
| rel-hm/user-churn | roc_auc | 0.7049 | 0.7061 |
| rel-stack/user-badge | roc_auc | 0.8921 | 0.8925 |
| rel-stack/user-engagement | roc_auc | 0.9065 | 0.9091 |
| rel-trial/study-outcome | roc_auc | 0.6957 | 0.7121 |
| rel-amazon/item-ltv | nmae | 0.0719 | 0.0726 |
| rel-amazon/user-ltv | nmae | 0.2429 | 0.2458 |
| rel-avito/ad-ctr | nmae | 0.3603 | 0.3786 |
| rel-event/user-attendance | nmae | 0.3163 | 0.3164 |
| rel-f1/driver-position | nmae | 0.3767 | 0.3972 |
| rel-hm/item-sales | nmae | 0.0757 | 0.0753 |
| rel-stack/post-votes | nmae | 0.1279 | 0.1362 |
| rel-trial/site-success | nmae | 0.7654 | 0.7517 |
| rel-trial/study-adverse | nmae | 0.1028 | 0.1074 |

Context sampling is stochastic and the selection arm picks its step against
sampled validation rows, so a fresh run will not match these digit for digit.
Pin the dataset revision recorded in
[`docs/downloads.md`](../../docs/downloads.md) and expect agreement to a few
tenths of a point.

### What it costs

Measured on our own runs, one GPU per task, all four stages, every attempt of
every job summed with restarts included: **135 GPU-hours for the 21 rt-j jobs
and 181 for the 21 rt-plurel jobs**. Per task that was 2.7-3.7 h on the three
rel-f1 tasks, 3-7 h on most, and 10-21 h on the tasks whose selection arm ran
to its 50k-step ceiling — rel-amazon/user-churn, rel-hm/user-churn,
rel-hm/item-sales, rel-stack/user-badge, and for rt-j also rel-amazon/user-ltv
and rel-amazon/item-ltv. The cards were 80 GB A100s and B200s, a B200 being
~2.5x an A100 per step. Run end to end in one process, a model is therefore
roughly a week of wall clock; in parallel it is as long as its slowest task.

`tokens_per_gpu` is the knob for a smaller card: the plan passes `2**17`, which
costs time rather than changing a result. A GPU is not optional — RT's
`flex_attention` path has no fused bfloat16 CPU kernel.

`collect.py` needs no GPU and runs in minutes.

### What has actually been run

- **Verified.** `plan.py` enumerates 42 jobs over the two models against an
  empty `RT_OUT_ROOT` and `0 jobs` against our finished outputs, at the exact
  stage paths `run.py` derives. `collect.py` gathered all 21 prediction tables
  for each model from those outputs, scored them through
  `relbench.submit.evaluate_submission` — reproducing every per-task number and
  both means in the tables above — and wrote the four validated
  `{classification,regression}.zip` packages.
- **Untested here.** The four stage bodies in `run.py` have not been re-run
  from this directory on a GPU. They are the code that produced the packages
  above, carried across with the cluster submission layer replaced by
  `plan.py`; we say untested because untested is what they are.

## Run it on your own data

Four things change, and nothing else has to.

1. **The preprocessed directory.** `RT_PRE_DIR` must be a RelBench-v3-format
   dataset preprocessed into `rt`'s own artifacts — a per-database directory
   with `table_info.json`, `nodes.rkyv`, `offsets.rkyv`, `p2f_adj.rkyv` and a
   `text_emb_all-MiniLM-L12-v2.bin`. [`pipelines/preprocess/`](../preprocess)
   is how to build one (that pipeline may not be in your checkout yet;
   [`docs/preprocess.md`](../../docs/preprocess.md) is the same ground). The
   embedder has to match the checkpoint's: `all-MiniLM-L12-v2`, `d_text=384`.
2. **The task list.** `TASKS` in [`plan.py`](plan.py) is a tuple of
   `(db, task)` pairs; pass your own as `tasks=` to `jobs()`. Each pair must
   name an entity-level classification or regression task with `train`, `val`
   and `test` splits in `table_info.json`. Nothing else is per-task: `run.py`
   reads `task_type` from the data and picks the loss (`bce` / `l1`) and the
   selection metric from it, and asserts rather than guessing if a stage leaves
   no output.
3. **The warm-start checkpoint.** `MODELS` in [`plan.py`](plan.py) maps a model
   name to a `load_ckpt_root` — a Hub id or a local directory holding
   `config.json` + `model.safetensors`. Add an entry to fine-tune from your own
   pretrained checkpoint; the model name is also the prefix of every stage
   directory, so two warm starts never collide. Pass `load_ckpt_root=None` to
   train from a random initialization under the same protocol, which turns
   `delta_finetune` off.
4. **The budgets, if your tasks are a different size.** Every one of them is an
   explicit argument of `jobs()`, and the values in `plan.py`'s `__main__` are
   ours: `selection_steps=50_000` with `patience_steps=10_000` (a task that
   hits the ceiling is the expensive case — lower both for a quick pass),
   `eval_freq=100` and `eval_rows=1024` for the selection sweep,
   `tune_rows=4096` for the context search, `test_ensemble_size=8`,
   `tokens_per_gpu` and `num_workers` for your card and host, `seed`.

The context grid itself is in `run.py` (`lcs_bw_pl_grid` and the
`ctx_size_list`), and is the one thing worth editing in place: a database whose
rows reach far fewer neighbors does not need `ctx=1024`, and a 60-shape search
is most of the `tune` stage's cost.

One constraint to know before you start, because it bites at the last stage and
not the first: the `test` stage emits its prediction table through
`rt.eval.relbench`, which loads the ground-truth test table with
`relbench.load_dataset(<your db's meta.json "source">).load_task(<task>)` and
raises if `meta.json` carries no `source`. So the first three stages are
data-generic, but the fourth needs your database and task to be loadable by the
`relbench` package under the name your preprocessed `meta.json` records. If
they are not, stop after the reporting arm and evaluate
`outer/latest_swa.safetensors` yourself with `rt.eval.main` on whatever split
you hold out.

`collect.py` likewise only scores and packages the canonical RelBench
leaderboard families, so on your own tasks there is nothing for it to
validate.
