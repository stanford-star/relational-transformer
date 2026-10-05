# Per-task fine-tuning

One checkpoint fine-tuned per RelBench task. Four stages: selection run on
`train`, context search, refit on `train + val`, 8-seed test ensemble. One GPU
per job, 21 jobs per model.

## Reproduce ours

```bash
export RT_PRE_DIR=data/relbench-preprocessed RT_OUT_ROOT=~/ckpts
pixi run python -m examples.finetune.plan    rt-j      # or rt-plurel, rt
pixi run python -m examples.finetune.collect rt-j      # scores, writes the zips
```

| model | warm start | clf roc_auc | reg nmae | cost |
|---|---|--:|--:|--:|
| `rt-j` | `stanford-star/rt-j` | **0.7902** | **0.2711** | 135 GPU-h |
| `rt-plurel` | `stanford-star/rt-plurel` | 0.7853 | 0.2757 | 181 GPU-h |
| `rt` | none | 0.7776 | 0.2927 | |

Warm-started models train an additive delta on frozen weights
(`delta_finetune`); `rt` trains the whole model. Outputs land under
`$RT_OUT_ROOT/no-wandb-entity/finetune/`. Reruns resume and skip finished
stages. `collect` reproduces the numbers above; the GPU stages were not re-run
from this tree.

## Run it on your own data

Change `RT_PRE_DIR`, the task list in [`plan.py`](plan.py), and the warm
start. The final stage scores through RelBench, so without a RelBench-loadable
`source` stop after `outer` and score the refit yourself.
