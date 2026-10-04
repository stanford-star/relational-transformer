# Per-task context tuning grid

A grid search over the context sampler's `(ctx, local_ctx_size, bfs_width,
prefer_latest)` on each task's validation split. It feeds three downstream
results: the tuned arm of the ensembling figure, the default-vs-tuned appendix
table ([`../valtest`](../valtest)), and the top-4-configurations leaderboard
ensemble ([`../leaderboard`](../leaderboard)).

**[`tuned_configs_rt-j.json`](tuned_configs_rt-j.json) is committed.** It is the output
of this stage for the paper's RT-J checkpoint, and every downstream stage reads
it from here, so you do not need to run the grid at all. Re-run it only to tune
a different checkpoint.

```bash
python -m reproduce.tune.plan       # 21 jobs, one GPU each; the critical path
python -m reproduce.tune.collect    # -> tuned_configs_rt-j.json
```

A second checkpoint's grid is the same code with a different `ckpt` and a
different `grid_stem`: [`tuned_configs_rt-plurel.json`](tuned_configs_rt-plurel.json)
is RT-PluRel's, produced that way, and
[`pipelines/icl`](../../pipelines/icl) is the entry point that runs this stage
and the leaderboard ensemble for either checkpoint. The two files are not
interchangeable — a context configuration is only valid for the checkpoint it
was tuned on — so each records the grid stem it was ranked from and every
consumer asserts it.

## Protocol

`ctx ∈ {512, 1024, 2048, 4096, 8192}` × `lcs ∈ {256, 512, 1024, 2048, 4096,
8192 | lcs ≤ ctx}` × `bw ∈ {8, 32, 128}` × `pl ∈ {True, False}` = **120
configurations per task**, 2520 in total. Each configuration is scored on the
task's validation split — 4096 rows at `shuffle_seed=0`, the prediction
averaged over 4 context seeds, `db_cutoff=None`. AUROC ranks classification
configurations, normalized MAE ranks regression.

A job is `rt.eval:main` in tune-only mode (`splits=["val"]`), one per task,
resumable per grid entry, writing `tuning.json` under
`$RT_OUT_ROOT/no-entity/tune/tune--<db>--<table>/`.

`tuned_configs_rt-j.json` holds, per task: the best configuration and its
validation score, the top-4 configurations, and the full 120-entry score table.
`collect.py` asserts the grid is complete and that `best_cfg` really is the top
score before writing.

## Cost

Measured on the paper's cluster: a task with 4096 validation rows is about
10.5h on an a100 and 4–6h on a b200; smaller validation splits scale down with
their row count (15 minutes for the rel-f1 tasks). Roughly 200 GPU-hours in
total, which is why the result is committed.
