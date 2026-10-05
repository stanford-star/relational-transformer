# Pretraining RT-J

Two phases: PluRel (86,211 tasks over 1,900 databases), then the Join (13,243
tasks over 523 databases) warm-started from phase 1.

## Reproduce ours

```bash
pixi run hf download stanford-star/plurel-preprocessed --repo-type dataset \
  --revision 9d70172425b44053f1270b19094ba6dcf7d66464 --local-dir data/plurel-preprocessed
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --revision ba5574ba659f0ef8b592793ed2ac9cb78dc87100 --local-dir data/the-join-preprocessed
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --revision 1016626ddb30c027b92458bf866903850cc205e1 --local-dir data/relbench-preprocessed
pixi run hf download stanford-star/plurel --repo-type dataset --local-dir data/plurel \
  --include "*/manifest.yaml" "*/scores.json"

pixi run python -m examples.preprocess.task_lists
CUDA_VISIBLE_DEVICES=0,1 pixi run python -m examples.pretrain.phases
```

The phases differ only in `load_ckpt_path`, `db_task_list` and `pre_dir`.
Phase 2 warm-starts from the released `stanford-star/rt-plurel`, so it also
runs on its own.

| phase | released as | file | step |
|---|---|---|--:|
| 1 PluRel | `stanford-star/rt-plurel` | `best_live_reg.safetensors` | 8,000 |
| 2 the Join | `stanford-star/rt-j` | `best_swa_clf.safetensors` | 9,000 |

Thousands of GPU-hours over many requeues. The phases were not re-run from
this tree.

## Run it on your own data

Point `pre_dir` and `db_task_list` in [`phases.py`](phases.py) at your data;
every argument is explicit there. Lower `total_steps` and `tokens_per_gpu` for
a smaller budget. [`mlock.py`](mlock.py) keeps the data in the page cache
between runs while you iterate ([`docs/train.md`](../../docs/train.md)).
