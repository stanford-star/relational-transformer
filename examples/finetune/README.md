# Fine-tuned leaderboard entries

One model fine-tuned per RelBench task. Four stages: selection run on
`train`, context search, refit on `train + val`, 8-seed test ensemble. One GPU
per job, 21 jobs per model.

## Reproduce the leaderboard entries

| entry | `<model>` | warm start | clf roc_auc | reg nmae | cost |
|---|---|---|--:|--:|--:|
| RT-J (fine-tuned) | `rt-j` | `stanford-star/rt-j` | **0.7902** | **0.2711** | 6.4 A100-h/task |
| RT-PluRel (fine-tuned) | `rt-plurel` | `stanford-star/rt-plurel` | 0.7853 | 0.2757 | 8.6 A100-h/task |
| RT | `rt` | none | 0.7776 | 0.2927 | |

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --revision 1016626ddb30c027b92458bf866903850cc205e1 --local-dir data/relbench-preprocessed

export RT_PRE_DIR=data/relbench-preprocessed RT_OUT_ROOT=~/ckpts
pixi run python -m examples.finetune.plan    <model>   # 21 tasks x 4 stages
pixi run python -m examples.finetune.collect <model>   # scores, writes the zips
```

`<model>` is `rt-j`, `rt-plurel` or `rt`. Warm-started models train an
additive delta on frozen weights (`delta_finetune`); `rt` trains the whole
model from random init. Outputs land under
`$RT_OUT_ROOT/no-wandb-entity/finetune/`. Reruns resume and skip finished
stages. `collect` reproduces the numbers above; the GPU stages were not re-run
from this tree.

Every stage trains and selects on `train` and `val` only, and every context is
built from rows dated strictly before the seed row's time.

## Run it on your own data

Change `RT_PRE_DIR`, the task list in [`plan.py`](plan.py), and the warm
start. The final stage scores through RelBench, so without a RelBench-loadable
`source` stop after `outer` and score the refit yourself.
