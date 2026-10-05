# Evaluating a checkpoint

One checkpoint, one context configuration, every task's test split. The
cheapest end-to-end run of a released model; the tuned and ensembled
leaderboard numbers are [`../icl`](../icl/).

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
a few hours; under `torchrun` rows shard across ranks.

`legacy` runs the earlier papers' nets at their published context against
`data/relbench-preprocessed/legacy` and writes one submission directory per
model under `~/ckpts/legacy/`.

## Run it on your own data

`job(...)` in [`plan.py`](plan.py) takes the checkpoint, `pre_dir`, task list
and context. Several `ctx_size_list` sizes or `lcs_bw_pl_grid` entries tune
per task on validation first; `test_ensemble_size` averages context seeds.
See [`docs/inference.md`](../../docs/inference.md).
