# Default vs tuned context

The appendix table comparing the shared default context `(8192, 256, 32,
prefer_latest=True)` against each task's tuned configuration, on the 8192-row
test subsample at a single context seed, with the selected
`(ctx, lcs, bw, pl)` columns.

**[`results.json`](results.json) is committed** — it is this table for the
paper's RT-J checkpoint.

Nothing is submitted here. Both columns are the ensemble-size-1 points of the
two curves in [`../enscurve`](../enscurve), which run the same 8192-row
subsample at exactly these two configurations, so the table is the n=1 point of
the ensembling figure read off the same runs.

```bash
python -m reproduce.valtest.collect   # -> results.json
```

`collect.py` asserts each curve's configuration and protocol — the default
tuple, or the task's `best_cfg` from `../tune/tuned_configs.json`, and
`items_per_task=8192`, `n_seeds=16` — before reading it.
