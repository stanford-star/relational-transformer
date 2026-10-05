# In-context leaderboard entries

A frozen checkpoint predicts a RelBench task with no gradient step on the
target database. Three stages: search each task's context on validation, run
the top 4 configs at 4 seeds on test, average and package.

## Reproduce the leaderboard entries

| entry | `<model>` | checkpoint | clf roc_auc | reg nmae | cost |
|---|---|---|---|---|---|
| RT-J (in-context) | `rt-j` | `stanford-star/rt-j` | 0.744512 | 0.323543 | ~200 GPU-h |
| RT-PluRel (in-context) | `rt-plurel` | `stanford-star/rt-plurel` | 0.729828 | 0.329467 | ~200 GPU-h |

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --revision 1016626ddb30c027b92458bf866903850cc205e1 --local-dir data/relbench-preprocessed
pixi run python -m examples.preprocess.task_lists

export RT_CKPT=stanford-star/rt-j RT_PLUREL_CKPT=stanford-star/rt-plurel
export RT_PRE_DIR=data/relbench-preprocessed RT_OUT_ROOT=~/ckpts RT_SHARE=~/ckpts/share
export RT_TASKS=data/db-task-lists/relbench-forecast.json

pixi run python -m examples.icl.tune     <model>   # 21 jobs; result is committed, skip
pixi run python -m examples.icl.collect  <model>   # -> configs/tuned_configs_<model>.json
pixi run python -m examples.icl.ensemble <model>   # 84 jobs
pixi run python -m examples.icl.package  <model>   # CSVs, scores, submission zips
```

`<model>` is `rt-j` or `rt-plurel`. One GPU per job. `tune` dominates the
cost; its output is committed under [`configs/`](configs/) so the other stages
run without it. The GPU stages were not re-run from this tree; the committed
configs and packaged CSVs are what we verified.

Every context is built from rows dated strictly before the seed row's time,
and no checkpoint was trained on any RelBench database.

## Run it on your own data

Point `RT_PRE_DIR` at your preprocessed data and `RT_TASKS` at a JSON file of
`[db, task]` pairs.
