# Evaluating a checkpoint

One checkpoint, one context configuration, every task's test split, scored by
RelBench's own evaluator. The cheapest end-to-end run of a released model; the
tuned-and-ensembled leaderboard numbers are [`../icl`](../icl/).

## Reproduce ours

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --revision 1016626ddb30c027b92458bf866903850cc205e1 --local-dir data/relbench-preprocessed

pixi run python -m examples.preprocess.task_lists      # -> data/db-task-lists/relbench-forecast.json
pixi run python -m examples.eval.plan        # RT-J, context (8192; 256, 32, latest)
pixi run python -m examples.eval.legacy      # RT-v1 and RT-PluRel, their papers' context
```

`plan` fetches `stanford-star/rt-j` from the Hub on demand and writes a RelBench
submission directory to `~/ckpts/no-wandb-entity/rt-eval/<run_id>/eval_out/`;
`pixi run python -m relbench.leaderboard <that dir>` re-scores it. One GPU, a
few hours over the 21 tasks; under `torchrun` the rows shard across ranks and
rank 0 scores them.

`legacy` runs the earlier papers' architectures (`rt.model.legacy`) at their
published context: ctx 1024, one BFS neighbourhood around the seed, no
random-walk tier, `bfs_width` 256 for RT-v1 and 128 for RT-PluRel. It reads
`data/relbench-preprocessed/legacy`, RelBench re-preprocessed with the
RT-v1-era boolean typing those nets' BCE heads expect, and writes one
submission directory per model under `~/ckpts/legacy/`. RT-v1 and the
`synth-real` PluRel arm load a per-task checkpoint; `synth` loads one
synthetic-only checkpoint for every task. Metrics reproduce the papers within
noise except RT-v1 on rel-avito, which degrades for sampler-level reasons
outside these configurations.

## Run it on your own data

`job(...)` in [`plan.py`](plan.py) takes the checkpoint, `pre_dir`, the task
list (`[db, task]` pairs inline or a JSON path) and the context; every other
argument of `rt.eval.main` is spelled out there. Pass several `ctx_size_list`
sizes or `lcs_bw_pl_grid` entries and the run tunes the context per task on
validation first; raise `test_ensemble_size` to average context seeds. The
knobs are documented in [`../../docs/inference.md`](../../docs/inference.md).
