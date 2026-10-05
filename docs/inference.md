# Inference

A trained checkpoint predicts every test row of a task from a sampled context,
with no gradient step. Predictions are scored with RelBench's own evaluator.

There is no CLI. [`examples/eval/plan.py`](../examples/eval/plan.py) calls
`rt.eval.main` with every argument spelled out; copy it and edit the call.

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed
pixi run python -m examples.preprocess.task_lists
pixi run python -m examples.eval.plan          # RT-J, 21 RelBench tasks, one GPU
```

The checkpoint comes from the Hub on demand. Under `torchrun` or one slurm
task per GPU, rows shard across ranks and rank 0 scores them.

## Tasks

`db_task_list` is a list of `(db, task)` pairs, inline or as a JSON path.
`rt.data.db_task_list(pre_dir, kinds)` enumerates what a preprocessed directory
ships; `kinds` is any of `"forecast"`, `"autocomplete"`. The quickest
end-to-end run is one task:

```python
main(load_ckpt_path="stanford-star/rt-j", pre_dir="data/relbench-preprocessed",
     db_task_list=[("rel-f1", "driver-top3")], ...)
```

## Output

Eval writes `<out_root>/<entity>/<project>/<run_id>/eval_out/`, one
`<db>__<task>.csv` per task, and prints per-task and mean metrics (AUROC for
classification, NMAE for regression). The directory is a valid RelBench
submission:

```bash
pixi run python -m relbench.submit eval_out     # re-score, write the zips
```

## Context

The context is the main quality knob. Arguments to `rt.eval.main`:

| argument | meaning | RT-J default |
|---|---|---|
| `ctx_size_list` | total context sizes in cells | `[8192]` |
| `lcs_bw_pl_grid` | `(local_ctx_size, bfs_width, prefer_latest)` sampler configs | `[(256, 32, True)]` |
| `num_walks`, `walk_length` | random walks that rank same-table neighbours | `10000`, `20` |

`local_ctx_size` caps the cells per BFS expansion around the seed row and
`bfs_width` the nodes kept per BFS level. `prefer_latest` picks the most recent
same-table rows over the most frequent. Extra sizes in `ctx_size_list` are
nearly free: every size is scored off a prefix of the largest context.

Give either list more than one entry and eval tunes per task on validation
before scoring test. `test_ensemble_size=N` then averages N context seeds on
test; `val_ensemble_size` does the same during tuning. Tuning writes
`tuning.json` beside `eval_out`. Pass `splits=["val"]` to tune without reading
test, and later `splits=["test"]` with the winning config as a one-entry grid.

```python
main(..., ctx_size_list=[4096, 8192],
     lcs_bw_pl_grid=[(256, 32, True), (512, 64, True)],
     val_ensemble_size=1, test_ensemble_size=4)
```

## Legacy checkpoints

RT-v1 and RT-PluRel keep their original architectures in `rt.model.legacy`.
`pixi run python -m examples.eval.legacy` evaluates both at their papers'
context against `data/relbench-preprocessed/legacy`, RelBench preprocessed
with the boolean typing those nets expect. Numbers match the papers within
noise, except RT-v1 on rel-avito.

## Optional: FAISS sampler

Seed selection by FAISS similarity is opt-in and needs cmake and a BLAS:

```bash
maturin develop --release --features vecdb
```
