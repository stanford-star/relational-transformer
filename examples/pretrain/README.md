# Pretrain

Self-supervised pretraining over the tasks in a `db_task_list`, with Muon +
AdamW, stochastic weight averaging, periodic validation on RelBench, and
checkpoint selection by mean validation metric. RT-J is two phases: PluRel
(86,211 tasks over 1,900 databases), then the Join (13,243 tasks over 523
databases) warm-started from phase 1.

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

`rt.train.main` takes every knob as an argument; a run is a script that calls
it. Copy [`phases.py`](phases.py) and edit: `pre_dir` holds the pretraining
databases and `eval_pre_dir` the validation ones. `db_task_list` is a JSON
list of `[db, task]` pairs; `rt.data.make_db_task_list(pre_dir, kinds)` enumerates
a directory and `rt.data.make_plurel_db_task_list` applies the PluRel filter.
Lower `total_steps` and `tokens_per_gpu` for a smaller budget.

## Outputs

Each run writes `<out_root>/<entity>/<project>/<run_id>/`:

| file | what |
|---|---|
| `steps=<N>.safetensors`, `swa_steps=<N>.safetensors` | live and SWA weights at an eval point |
| `latest.safetensors`, `latest_swa.safetensors` | the most recent of each |
| `best_{live,swa}_{clf,reg}.safetensors`, `best_{clf,reg}.safetensors` | selected by mean validation AUROC / NMAE |
| `resume.pt` | full state, rewritten every eval and every `resume_save_mins` |

With `keep_all_ckpts=False` only the current best checkpoints are kept.
Relaunching with the same `run_id` resumes from `resume.pt`, on any number of
GPUs. A checkpoint loads with `rt.model.load_rt_model(path)`, same as a Hub
one.

## Knobs worth knowing

- `swa_momentum`: momentum of the weight average kept beside the live net. `None` disables SWA. Fixed across a resume.
- `optimizer`: `"muon"` (hidden matrices to Muon, the rest to AdamW; the released setting) or `"adamw"`. Fixed across a resume.
- `wd`: decays every weight matrix and never a gain or bias.
- `lr_warmup_steps`, `lr_decay_steps`: linear warmup from 0, linear decay to 0 over the last steps of `total_steps`. `0` disables either.
- `early_stop_after_steps`: stop once neither net improved its val metric for that many steps. Needs `"val"` in `eval_splits`.
- `can_select_init_model`: whether the step-0 eval may win selection. `False` for warm starts.
- `delta_finetune`: train a zero-initialised additive delta on frozen loaded weights, so weight decay pulls toward the pretrained point. Needs `load_ckpt_path`.
- `train_splits`: `["train", "val"]` trains on validation labels too; drop `"val"` from `eval_splits` then.
- `db_cutoff`: trim the database to `"val"` or `"test"` time before building contexts.
- `eval_ensemble_size`: average the in-loop eval over that many context seeds.
- `mmap_populate`: load the data into the page cache at startup. `stage_dir` instead copies `pre_dir` and `eval_pre_dir` to a node-local directory first; use it when the shared filesystem evicts the working set.

## Multi-GPU

One process per GPU with `RANK`, `LOCAL_RANK` and `WORLD_SIZE` set; the model
is replicated, not sharded. `torchrun --standalone --nproc-per-node=auto -m
examples.pretrain.phases` on one node, or under slurm one task per GPU:

```bash
srun --ntasks-per-node=8 --gres=gpu:8 pixi run python -m examples.pretrain.phases
```

with `SLURM_PROCID`/`SLURM_LOCALID`/`SLURM_NTASKS` exported as the torch
names. Any launcher that does this works; RT has no cluster code of its own.

## Iterating locally

Each run repopulates the page cache at startup. To skip that while iterating,
hold the data resident once with [`mlock.py`](mlock.py) and train with
`mmap_populate=False`. `mlock` needs a high `RLIMIT_MEMLOCK` (`ulimit -l unlimited`).
