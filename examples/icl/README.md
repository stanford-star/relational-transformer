# In-context

A frozen checkpoint predicts a RelBench task with no gradient step on the
target database. Three stages: search each task's context on validation, run
the top 4 configs at 4 seeds on full test, average and package.

## Reproduce ours

```bash
export RT_CKPT=...          # or RT_PLUREL_CKPT for the rt-plurel arm
export RT_PRE_DIR=... RT_OUT_ROOT=... RT_SHARE=...
export RT_TASKS="$(pixi run python -c \
  'from rt.data import get_mixture_path; print(get_mixture_path("relbench", "forecast"))')"

pixi run python -m examples.icl.tune     rt-plurel   # 21 jobs; skip, result is committed
pixi run python -m examples.icl.collect  rt-plurel   # -> reproduce/tune/tuned_configs_rt-plurel.json
pixi run python -m examples.icl.ensemble rt-plurel   # 84 units (21 tasks x 4 configs)
pixi run python -m examples.icl.package  rt-plurel   # CSVs + the packaging command
```

Swap `rt-plurel` for `rt-j`. One GPU per job.

| entry | clf roc_auc | reg nmae | cost |
|---|---|---|---|
| RT-J (in-context) | 0.744512 | 0.323543 | ~200 GPU-h |
| RT-PluRel (in-context) | 0.729828 | 0.329467 | ~200 GPU-h |

Stage 1 dominates the cost (~10.5 h per task on an A100, ~4 h on a B200);
stage 2 is minutes to hours per unit. The GPU stages are untested from this
tree — the committed configs and the packaged CSVs are what we verified.

## Run it on your own data

Point `RT_PRE_DIR` at your preprocessed data ([`../preprocess`](../preprocess/))
and `RT_TASKS` at a JSON file of `[db, task]` pairs. Nothing else changes.
