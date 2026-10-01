# Context-ensembling curves

The test-time-compute figure: metric against the number of ensembled context
seeds, for the shared default context `(8192, 256, 32, prefer_latest=True)` and
for each task's tuned context from [`../tune`](../tune); plus its per-task
appendix figures. The two variants are also the two columns of
[`../valtest`](../valtest)'s table, read at ensemble size 1.

```bash
python -m reproduce.enscurve.plan     # 21 jobs per variant, one GPU each
python -m reproduce.enscurve.reduce   # -> ../series/enscurve/{default,tuned}.json
```

The tuned variant reads the committed `../tune/tuned_configs.json`, so neither
variant waits on the grid.

## Protocol

Per (variant, task): 16 independent context seeds (`rt.eval`'s ensemble seed
family off base seed 0) at the variant's fixed configuration, on the fixed
8192-row test subsample (`shuffle_seed=0`), `db_cutoff=None`. Raw per-row
predictions are averaged over the first k seeds and scored on the normalized
scale at every k = 1…16 — the same quantity `rt.eval`'s ensembling scores, not
a mean of per-seed scores. A job resumes per seed from
`<db>__<table>.state.npz`, and asserts that every seed scored the same rows in
the same order before summing.

Curves land at `$RT_OUT_ROOT/enscurve/<variant>/<db>__<table>.json`.

## Cost

Measured on the paper's cluster: about 1h05 for a 16-seed curve over 8192 rows
on an a100, whatever the database, and under 20 minutes for tasks with fewer
test rows than that. Both variants together are 42 jobs.
