# Pretraining RT-J

Two phases: PluRel (86,211 tasks over 1,900 databases), then the Join (13,243
tasks over 523 databases under a 5 GB per-database cutoff) warm-started from
phase 1. 85,562,530 parameters; 2 GPUs per phase.

## Reproduce ours

```bash
pixi run hf download stanford-star/plurel-preprocessed --repo-type dataset \
  --revision 9d70172425b44053f1270b19094ba6dcf7d66464 --local-dir data/plurel-preprocessed
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --revision ba5574ba659f0ef8b592793ed2ac9cb78dc87100 --local-dir data/the-join-preprocessed
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --revision 1016626ddb30c027b92458bf866903850cc205e1 --local-dir data/relbench-preprocessed

CUDA_VISIBLE_DEVICES=0,1 pixi run python -m pipelines.pretrain.phases
```

The two phases are the same call differing in three arguments —
`load_ckpt_path`, `db_task_list`, `pre_dir`. Phase 2's warm start is the
released `stanford-star/rt-plurel`, so it runs standalone.

| phase | released as | file the run wrote | step | selected by |
|---|---|---|--:|---|
| 1 PluRel | `stanford-star/rt-plurel` | `best_live_reg.safetensors` | 8,000 | val nMAE |
| 2 the Join | `stanford-star/rt-j` | `best_swa_clf.safetensors` | 9,000 | val AUROC 74.36 |

Neither phase stops at that step; the run's own selection writes those files.
Thousands of GPU-hours, spread over requeues — there is no single wall clock.
Job arguments were checked against `rt.train.main`'s signature; the phases
themselves were not re-run.

## Run it on your own data

```python
from rt.data import get_mixture_path, list_mixtures

list_mixtures()                            # every (collection, name)
get_mixture_path("the-join", "forecast")   # or a path to your own JSON
```

Your mixture is a JSON list of `[db, task]` pairs. For a smaller budget, lower
`total_steps` and `tokens_per_gpu` in [`phases.py`](phases.py); every argument
is explicit there.
