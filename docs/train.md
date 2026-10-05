# Pretrain

Self-supervised pretraining over the tasks in a `db_task_list`, with Muon +
AdamW, stochastic weight averaging, periodic validation on RelBench, and
checkpoint selection by mean validation metric.

There is no CLI. `rt.train.main` takes every knob as an argument; a run is a
script that calls it. Copy
[`examples/pretrain/phases.py`](../examples/pretrain/phases.py) and edit.

```bash
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed
pixi run python -m examples.preprocess.task_lists
CUDA_VISIBLE_DEVICES=0 pixi run python -m examples.pretrain.phases
```

## Data

`pre_dir` holds the pretraining databases and `eval_pre_dir` the validation
ones; both are local directories ([downloads.md](downloads.md)). `db_task_list`
is a JSON list of `[db, task]` pairs; `rt.data.db_task_list(pre_dir, kinds)`
enumerates a directory and `rt.data.plurel_train_db_task_list` applies the
PluRel filter. A task the build cannot predict is reported and skipped.

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
GPUs.

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
- `mmap_populate`: load the data into the page cache at startup. `stage_dir` instead copies `pre_dir` and `eval_pre_dir` to a node-local directory first (env vars expanded, one rank per node copies, requeues reuse it); use it when the shared filesystem evicts the working set.

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
hold the data resident once with
[`examples/pretrain/mlock.py`](../examples/pretrain/mlock.py) and train with
`mmap_populate=False`. `mlock` needs a high `RLIMIT_MEMLOCK` (`ulimit -l unlimited`).

## Loading checkpoints

```python
from rt.model import load_rt_model
model, config = load_rt_model("~/ckpts/run1/best_clf.safetensors", device="cuda")
model, config = load_rt_model("stanford-star/rt-j")
```
