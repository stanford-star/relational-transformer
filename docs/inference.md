# Inference

Run a trained RT checkpoint on preprocessed data: load the checkpoint, sample an
in-context context for each test row, run the model's forward pass, and (for
RelBench) score the predictions with RelBench's own evaluator. There is no
fine-tuning — RT predicts zero-shot from the context it is given.

A checkpoint is a local path or a Hub model repo such as
`stanford-star/rt-j` — a single checkpoint handles classification and
regression tasks alike. All tasks in the configured task list are evaluated.

## Prerequisite: preprocessed data

Inference takes a `pre_dir` of preprocessed data: a **local directory**,
either produced by you (see [preprocess.md](preprocess.md)) or downloaded up
front (see [downloads.md](downloads.md)). Data is never fetched on demand;
checkpoints still are. To reproduce the RelBench numbers:

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed

# the checkpoint still comes from the Hub, on demand
pixi run python examples/eval.py
```

There is no CLI: [`examples/eval.py`](../examples/eval.py) calls `rt.eval.main`
with every argument spelled out (`load_ckpt_path="stanford-star/rt-j"`,
`pre_dir="data/relbench-preprocessed"`, ...). Copy it and edit the call.

## Inference with default context

The example above runs **simple** inference: one default context config
(`lcs_bw_pl_grid=[(256, 32, True)]`, total `ctx_size_list=[8192]`) on the
test split of every task in the default task list
(`data/relbench-preprocessed/db-task-lists/forecast.json`, the 21-task RelBench
benchmark). For each test row the sampler builds a context (a sampled
neighborhood of the relational graph), the model does a single forward pass,
and predictions are keyed back to each row by its seed node index. Because that
key is the seed node index and not a row position, per-row predictions stay
aligned regardless of eval row order.

Eval runs single-process on one GPU by default. For multi-GPU eval, launch it
under torchrun:

```bash
srun --ntasks-per-node=8 --gres=gpu:8 pixi run python examples/eval.py
```

Rows are sharded across ranks and gathered back on rank 0, which scores them
and writes the submission CSVs; the results are identical to a single-GPU run.

**Your own database (not RelBench).** `rt.eval` is wired to the RelBench
benchmark, but the pieces compose directly for any database — prediction is just
the model's forward over a sampled context. The [fully worked Colab
notebook](../byod/colab.ipynb) walks through it end-to-end (DuckDB
database → tasks defined in SQL → preprocess → forward pass → map outputs →
score) on the released RT-J checkpoints.

## Inference on a subset of tasks

The task set is `db_task_list`: `(db, task)` pairs given inline or as a path to
a JSON file of pairs. To run one task:

```python
main(load_ckpt_path="stanford-star/rt-j",
     pre_dir="data/relbench-preprocessed",
     db_task_list=[("rel-f1", "driver-top3")], ...)
```

That reads just that task's data out of `pre_dir`, so it's the quickest
way to try the model end-to-end — and you can fetch only that database:
`hf download stanford-star/relbench-preprocessed --repo-type dataset --local-dir
data/relbench-preprocessed --include "rel-f1/*" "db-task-lists/*"`. Curated
lists ship with the data:
`<pre_dir>/db-task-lists/{forecast,autocomplete,all}.json`.

## Evaluate with the RelBench evaluator

Eval writes a valid RelBench **submission directory** to
`<out_root>/<entity>/<project>/<run_id>/eval_out`: one
`<dataset>__<task>.csv` prediction table per task, scored through **RelBench's own
leaderboard evaluator** (`relbench.leaderboard`). Eval denormalizes regression
predictions to the original target scale (`y = pred*std + mean`, train-split
stats), maps classification logits to probabilities (sigmoid), and keys each
prediction to its relbench `(entity_col, time_col)`. It prints per-task and mean
test metrics — **AUROC** for clf, **NMAE** for reg.

Re-validate / re-score a submission dir any time, and submit it via the [RelBench
leaderboard procedure](https://relbench.stanford.edu):

```bash
pixi run python -m relbench.leaderboard eval_out
```

## Context engineering

Because RT predicts from context alone, the **context sampled for each row** is
the main quality knob. All are arguments to `rt.eval.main`:

| argument | meaning | released value |
|---|---|---|
| `ctx_size_list` | total context sizes (cells) to score; several = tune the size per task, and they share one pass | `[8192]` |
| `lcs_bw_pl_grid` | `(local_ctx_size, bfs_width, prefer_latest)` sampler configs; one entry = use it directly, several = tune per task on validation | `[(256, 32, True)]` |
| `num_walks` | random walks used to rank same-table neighbors | 10000 |
| `walk_length` | max length of each random walk | 20 |

Within a grid entry, larger `local_ctx_size` (max cells per BFS expansion
around the seed) and `bfs_width` (max DB nodes kept per BFS level) pull more
relational neighborhood into each row's context (more signal, more tokens);
`ctx_size_list` caps the total, and a `local_ctx_size` above a given size is
dropped from the grid: it is not distinct from `local_ctx_size == ctx_size`.
Adding sizes to `ctx_size_list` is nearly free -- each item's context is built
once at the largest and every size is scored off a prefix of it, so a whole
list is one pass. `prefer_latest` controls *which* same-table
neighbors win that budget — the most recent rows (`True`, default) or the most
frequent (`False`). The best setting is task-dependent — which motivates tuning
and ensembling below.

`shuffle_seed` seeds the per-task subset selection and item
shuffle. Fixing it while capping rows with `val_items_per_task=N` evaluates
the *same* N validation rows across every config — the basis for a
like-for-like context grid search.

## Context tuning

Rather than fix one config, **tune** the context per task: pass several
`lcs_bw_pl_grid` entries or `ctx_size_list` sizes and eval scores each
combination on the **validation** split,
keeping the best per task before scoring test (here with a single test seed, so
no averaging yet):

```python
main(load_ckpt_path="stanford-star/rt-j",
     pre_dir="data/relbench-preprocessed",
     ctx_size_list=[4096, 8192],
     lcs_bw_pl_grid=[(256, 32, True), (512, 64, True)],
     val_ensemble_size=1, test_ensemble_size=1, ...)
```

## Context ensembling

Context sampling is stochastic, so averaging predictions over several context
**seeds** reduces variance. Set `test_ensemble_size=N` (> 1): the per-task
tuned config runs with N independent context seeds on test and the per-row
predictions are averaged before scoring:

```python
main(load_ckpt_path="stanford-star/rt-j",
     pre_dir="data/relbench-preprocessed",
     ctx_size_list=[4096, 8192],
     lcs_bw_pl_grid=[(256, 32, True), (512, 64, True)],
     val_ensemble_size=1, test_ensemble_size=4, ...)
```

The average is scored after every seed, so the log carries the metric at every
ensemble size from 1 to N; the submission CSVs are the full ensemble's.

`val_ensemble_size` is the same knob on the tuning side: each config in the
grid is ranked on its prediction averaged over that many seeds. Leave it at 1
to tune cheaply, or match `test_ensemble_size` to rank each config at the
quantity it will actually be used at.

Tuning (on validation) and ensembling (on test) engage automatically whenever
the grid has more than one entry or `test_ensemble_size` exceeds 1: pick the
best context config per task, then average that config over the seeds.

## Tuning without touching test

Tuning writes `tuning.json` beside `eval_out` — per task, every config's
validation score, the winning config and its value. Drop `"test"` from
`splits` to stop there, reading no test data at all:

```python
main(load_ckpt_path="stanford-star/rt-j",
     pre_dir="data/relbench-preprocessed",
     splits=["val"],
     ctx_size_list=[4096, 8192],
     lcs_bw_pl_grid=[(256, 32, True), (512, 64, True)],
     val_ensemble_size=1, test_ensemble_size=1, ...)
```

A later run then evaluates that decision, passing the recorded winner as a
one-entry grid with as many seeds as you want:

```python
main(..., splits=["test"], ctx_size_list=[8192], lcs_bw_pl_grid=[(256, 32, True)],
     val_ensemble_size=1, test_ensemble_size=4)
```

## Optional: FAISS vector-DB sampler

The default sampler is FAISS-free. The opt-in FAISS vector-db sampler (for
nearest-neighbor context retrieval) is built manually and additionally needs
cmake + a BLAS:

```bash
maturin develop --release --features vecdb
```

## Legacy checkpoints (RT-v1, RT-PluRel)

The released checkpoints of the earlier papers use their original
architectures, kept verbatim in `rt.model.legacy` (state-dict compatible with
the published `.pt` files). [`examples/eval_legacy.py`](../examples/eval_legacy.py) reproduces the published
context configuration (ctx 1024, one BFS neighborhood around the seed,
bfs_width 256, no random-walk tier) and writes RelBench leaderboard submission
dirs:

```python
from examples.eval_legacy import eval_plurel, eval_v1

eval_v1()                      # RT-v1 (ICLR 2026), task-wise checkpoints
eval_plurel(mode="synth")      # RT-PluRel (ICML 2026), one synthetic-only checkpoint
eval_plurel(mode="synth-real") # ... or the task-wise continued-pretraining ones
```

`synth` uses the best synthetic-only pretraining checkpoint (same for
all tasks); `synth-real` uses the task-wise continued-pretraining
checkpoints. All three are in-context: no checkpoint ever trained on the
target task's database (v1, synth) or task (synth-real).

`data/relbench-preprocessed/legacy` (from
`stanford-star/relbench-preprocessed`, subdir `legacy/`) holds RelBench re-preprocessed
with the RT-v1-era boolean-typing
rules (binary targets and a few db columns become a real Boolean semantic
type instead of z-scored numbers) before the regular pipeline. The legacy
nets read classification targets from their BCE-trained boolean head, so this
is the data they need; `examples/eval_legacy.py` points at it. `eval_plurel`
additionally uses the paper's bfs_width 128. Metrics reproduce the papers within noise except
RT-v1 on rel-avito, which degrades for sampler-level reasons outside these
configs.
