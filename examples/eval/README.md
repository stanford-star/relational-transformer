# Evaluate

A trained checkpoint predicts every test row of a task from a context sampled
around it, with no gradient step. One checkpoint, one context configuration,
every task's test split. The cheapest end-to-end run of a released model; the
tuned and ensembled leaderboard numbers are [`../icl`](../icl/).

## Reproduce ours

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --revision 1016626ddb30c027b92458bf866903850cc205e1 --local-dir data/relbench-preprocessed

pixi run python -m examples.preprocess.task_lists
pixi run python -m examples.eval.plan        # RT-J
pixi run python -m examples.eval.legacy      # RT-v1 and RT-PluRel
```

`plan` fetches `stanford-star/rt-j` on demand and writes a RelBench submission
directory under `~/ckpts/no-wandb-entity/rt-eval/<run_id>/eval_out/`. One GPU,
a few hours; under `torchrun` or one slurm task per GPU, rows shard across
ranks and rank 0 scores them.

`legacy` runs the earlier papers' nets (`rt.model.legacy`) at their published
context against `data/relbench-preprocessed/legacy`, RelBench preprocessed with
the boolean typing those nets expect, and writes one submission directory per
model under `~/ckpts/legacy/`. Numbers match the papers within noise, except
RT-v1 on rel-avito.

## Run it on your own data

`job(...)` in [`plan.py`](plan.py) calls `rt.eval.main` with the checkpoint,
`pre_dir`, task list and context. `db_task_list` is a list of `(db, task)`
pairs, inline or as a JSON path; `rt.data.db_task_list(pre_dir, kinds)`
enumerates what a preprocessed directory ships, `kinds` any of `"forecast"`,
`"autocomplete"`. The quickest end-to-end run is one task:

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
[`../ctx_viz.py`](../ctx_viz.py) shows what a configuration pulls in.

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

## Optional: FAISS sampler

Seed selection by FAISS similarity is opt-in and needs cmake and a BLAS:

```bash
maturin develop --release --features vecdb
```
