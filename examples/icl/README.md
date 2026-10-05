# In-context

A frozen checkpoint predicts a RelBench task with no gradient step. Three
stages: search each task's context on validation, run the top 4 configs at 4
seeds on test, average and package.

## Reproduce ours

```bash
export RT_CKPT=...          # or RT_PLUREL_CKPT for rt-plurel
export RT_PRE_DIR=... RT_OUT_ROOT=... RT_SHARE=...
export RT_TASKS=data/db-task-lists/relbench-forecast.json

pixi run python -m examples.icl.tune     rt-plurel   # 21 jobs; result is committed
pixi run python -m examples.icl.collect  rt-plurel   # -> configs/tuned_configs_rt-plurel.json
pixi run python -m examples.icl.ensemble rt-plurel   # 84 jobs
pixi run python -m examples.icl.package  rt-plurel   # CSVs + the packaging command
```

Swap `rt-plurel` for `rt-j`. One GPU per job.

| entry | clf roc_auc | reg nmae | cost |
|---|---|---|---|
| RT-J (in-context) | 0.744512 | 0.323543 | ~200 GPU-h |
| RT-PluRel (in-context) | 0.729828 | 0.329467 | ~200 GPU-h |

`tune` dominates the cost. The GPU stages were not re-run from this tree; the
committed configs and packaged CSVs are what we verified.

## Run it on your own data

Point `RT_PRE_DIR` at your preprocessed data and `RT_TASKS` at a JSON file of
`[db, task]` pairs.
