# Pretrain

Self-supervised pretraining of a Relational Transformer over the tasks in
`db_task_list` (the Join). Includes Muon+AdamW
optimization, stochastic weight averaging (SWA), periodic validation against
RelBench, checkpointing, and automatic selection of the best clf / reg checkpoint
by mean validation metric.

Checkpoints land in the per-run directory
`<out-root>/<entity>/<project>/<id>/` as `steps=<N>.safetensors` (live) and
`swa_steps=<N>.safetensors` (SWA, when `swa_momentum` is set), with
`latest.safetensors` (and `latest_swa.safetensors`) pointing at the most recent
of each -- a stable path for a reader that wants the current weights of a run
still training, or of one with no val split and so no `best_*`. At the end the
run copies each net's best
classifier and regressor to `best_live_clf.safetensors` /
`best_swa_clf.safetensors` (and the `reg` pair), plus `best_clf.safetensors` /
`best_reg.safetensors` for whichever of the two nets won. Selection is by mean
**validation** metric only -- AUROC for clf, NMAE for reg -- so a test split
evaluated alongside never picks a checkpoint. With
`keep_all_ckpts=False` only the checkpoints the best-so-far still points at are
kept -- the rest are deleted as they are superseded, and the latest weights stay
available in `resume.pt` (rewritten at every eval and once more at the end). Multi-GPU is
automatic under `torchrun`, and a run relaunched with the same `run_id`
resumes automatically from `resume.pt` in that same directory
(preemption-safe).

## Prerequisite: preprocessed data

Pretraining takes a `pre_dir` of preprocessed pretraining data (the
Join) and an `eval_pre_dir` of preprocessed RelBench for validation. Both are
**local directories** — either produced by [preprocess.md](preprocess.md) or
downloaded up front; nothing is fetched on demand (see
[downloads.md](downloads.md) for how to fetch a subset):

```bash
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed
```

Those two paths are what [`examples/pretrain/phases.py`](../examples/pretrain/phases.py)
passes for phase 2 (`pre_dir="data/the-join-preprocessed"`,
`eval_pre_dir="data/relbench-preprocessed"`). The preprocessed Join is large, so on
a cluster fetch it **once** to shared storage and point every run at that path.

The tasks to train on are given by `db_task_list` — `(db, task)` pairs as a
JSON file. Names resolve against the tasks the db ships (recorded in its
`meta.json`); one the build cannot predict — a recommendation task, or an
entry left over from an older build — is reported and ignored, not fatal. The lists are generated from the downloaded data by
`rt.data.db_task_list(pre_dir, kinds)`: `kinds=("forecast",)` is every forecast
task, `("autocomplete",)` every `kind: autocomplete` task (predict a column of a
db table, train-split only), and the default (both) over
`data/the-join-preprocessed` is exactly RT-J's phase-2 pretraining list: 13,243
pairs over 523 databases. RT-J's phase-1 list is
`rt.data.plurel_train_db_task_list(pre_dir, raw_dir)`: PluRel databases
`plurel-3000`… in order, dropping any whose table has more than 5 foreign keys,
first 1,900 survivors; a column is a task if it is not a source node of the
generating graph, has at least 2 distinct values, and (classification) has at
least 2 classes with a majority class of at most 99%, or (regression) a standard
deviation of at least 1e-4 — 86,211 pairs. The per-column statistics come from
`scores.json` beside each database's `manifest.yaml` in the raw
`stanford-star/plurel` repo. `python -m examples.preprocess.task_lists` writes both lists
plus RelBench's forecast list under `data/db-task-lists/`.

`train_splits` picks which splits of those tasks the training stream draws
from. `["train"]` is the usual choice; `["train", "val"]` fine-tunes on the
validation labels too, which means `eval_splits` must drop `"val"` — a split
that is trained on cannot select the checkpoint, and with no val metric the
final step is what the run keeps (and `early_stop_after_steps` must be `None`).

`swa_momentum` is the momentum of the running weight average kept beside the
live net, evaluated, checkpointed and selected on in parallel with it. `None`
turns SWA off: no second net is built, so there are no `swa_steps=` or
`best_swa_*` files and nothing is selected on a `swa/` metric. It cannot be
changed across a resume — the average is part of `resume.pt`.

`wd` decays every weight matrix — the hidden matrices Muon holds and the
encoders'/decoders' alike — and never a gain or a bias. That choice is made on
its own, not by which optimizer a parameter went to: Muon takes hidden matrices
only, but decay is about shape, so the two splits are not the same one.

`delta_finetune` trains a zero-initialized additive delta on frozen pretrained
weights rather than the weights themselves. The gradient of the delta is the
gradient of the weight, and Muon's lr scaling depends on shape alone, so the
update is unchanged — what moves is the point weight decay pulls to: the
pretrained weights instead of zero. It needs a `load_ckpt_path`, and with
`wd=0` it is exactly ordinary fine-tuning.

`db_cutoff` trims the database the contexts are built from to a split's
timestamp: `"test"` (relbench's `get_db(upto_test_timestamp=True)`), `"val"`
one split earlier, or `None` for no trim. Evaluate a split under its own
cutoff — a val metric scored against a test-cutoff database sees the rows the
val labels postdate, and stops predicting test.

`eval_ensemble_size` averages the in-loop eval over that many context seeds
before scoring — the **predictions** are averaged per item, as
`rt.eval.run_ensemble` does, not the per-seed scores. `1` is the ordinary
single-context eval and uses `eval_context_seed` directly; above that the seeds
are the same mixed family `rt.eval` sweeps, so member *i* matches. Each member
is one more evaluator, built once and reused at every eval point: one more pass
over the eval split per eval, and one more set of loader workers resident for
the run.

`optimizer` selects `"muon"` — hidden weight matrices to Muon, the per-sem-type
encoders/decoders and every 0/1-D parameter to AdamW, which is what the released
checkpoints were trained with — or `"adamw"`, one AdamW over all of them with the
same weight-decay split. It cannot be changed across a resume: the optimizer
state in `resume.pt` is per optimizer.

`lr_warmup_steps` and `lr_decay_steps` shape the learning rate: linear warmup
from 0 over the first, linear decay to 0 over the last that many steps of
`total_steps`, and `0` disables either end. The decay is measured back from
`total_steps`, so a run that ends before it — early stopping, or a
`total_steps` set far beyond where the run is meant to stop — never reaches the
decay at all.

`early_stop_after_steps` ends the run early once neither the live nor the SWA
val metric has improved on — or matched again — its best for that many steps,
checked at each eval. `None` runs the full `total_steps`. It needs `"val"` in
`eval_splits`; with nothing selected there is nothing to stop on.

Every run evaluates at step 0 as well, so the starting point is on the curve.
`can_select_init_model` says whether that eval may also pick the best
checkpoint; `False` logs it and leaves it out of the selection and the
early-stopping patience — the usual choice for a warm start, where eval noise
could otherwise hand the win to the weights the run began with.

## Running a training script

There is no CLI. `rt.train._train` is a function that takes every knob as a
required argument; a run is a script that calls it. Copy
[`examples/pretrain/phases.py`](../examples/pretrain/phases.py) and edit what you want.
`pixi install` builds the rustler sampler as part of the environment; nothing
else to build.

`phase_two` there is **one** run: RT-J's second phase, on the Join, warm
started from the released phase-1 checkpoint
(`load_ckpt_path="stanford-star/rt-plurel"`). RT-J is two phases — PluRel first,
then the Join from those weights — so reproducing it is both, in order; the
[`examples/pretrain/`](../examples/pretrain) README records which step of which
phase each released checkpoint is and how it was selected.

```bash
CUDA_VISIBLE_DEVICES=0 pixi run python -m examples.pretrain.phases    # one GPU
```

## Multi-GPU single-node training

One process per GPU, each told who it is; the model is replicated per rank (full
model + optimizer on every rank, no sharding). Under slurm that is one line:

```bash
srun --ntasks-per-node=8 --gres=gpu:8 pixi run python -m examples.pretrain.phases
```

All RT needs is `RANK`/`LOCAL_RANK`/`WORLD_SIZE` in each task's environment, so
export them from slurm's `SLURM_PROCID`/`SLURM_LOCALID`/`SLURM_NTASKS` — and
because each rank is a slurm task, a preemption signal reaches all of them.
Outside slurm, `torchrun --standalone --nproc-per-node=auto -m examples.pretrain.phases`
works the same way.

Give the process as much of the node's RAM as you can: by default each run
populates the preprocessed data into the page cache at startup
(`mmap_populate=True`) so the GPUs are fed instead of cold-faulting
the (large) data from shared storage per item.

That only helps if the page cache keeps the data. On a shared filesystem
whose client evicts it -- Lustre's lock LRU and client cache cap do, for a
working set of a few hundred GB -- pass `stage_dir`: a node-local directory
that `pre_dir` and `eval_pre_dir` are copied into before
anything is opened (`rt.data.stage_paths`). Environment variables are
expanded, so `stage_dir="$TMPDIR/hf"` names the job's own scratch without
knowing the job id; one rank per node copies (a few hundred GB takes about
two minutes over a fast interconnect, the top-level directories in parallel),
the others wait, and a `.staged` marker makes a requeue onto the same node
skip the copy. `stage_dir=None` reads in place, which is right where the
shared filesystem keeps the working set resident.

## On a cluster

RT has no launcher of its own and no cluster-specific code. It trains under any
launcher that gives each process `RANK`, `LOCAL_RANK` and `WORLD_SIZE`, with one
process per GPU: `torchrun` on a single node, and under slurm one task per GPU,

```bash
srun --ntasks-per-node=8 --gres=gpu:8 pixi run python -m examples.pretrain.phases
```

with the task's `SLURM_PROCID`/`SLURM_LOCALID`/`SLURM_NTASKS` exported as the
torch names. Because each rank is a slurm task, a preemption signal reaches all
of them. Anything else your site uses to start one process per GPU works too;
`pre_dir` and `eval_pre_dir` just have to name a path every node can read.

If you would rather not write that plumbing,
[`roach`](https://github.com/rishabh-ranjan/roach) is one option: it does the
`SLURM_*`-to-`RANK` translation, refuses a dirty or unpushed tree, records the
commit, checks your arguments against the target's signature, and hands slurm a
script that clones that commit, builds the environment on the node, and starts
one rank per GPU.

```python
from roach.slurm import Resources, submit

from examples.pretrain.phases import phase_two

submit("rt.train:main",
       args=phase_two(load_ckpt_path="stanford-star/rt-plurel", pre_dir=..., ...).args,
       resources=Resources(...),
       cluster=...,
       name="rt-j",
       repo_root=..., log_root=..., clone_root=..., secrets_dir=...)
```

**Resume** is automatic from `$OUT_DIR/resume.pt` and **GPU-count flexible**: a
run preempted on 4×8 GPUs can resume on a single 4-GPU node with the same
`OUT_DIR` — the data stream is re-seeded by the resumed step, so nothing is
replayed and determinism holds across the world-size change. A time-based dump
every `resume_save_mins` minutes (20 in the examples) bounds lost progress.

## Avoiding data loading during debug iterations

By default each run re-populates the preprocessed data into RAM at startup. When
iterating on training code, that reload is wasted work on every restart. Lock the
data into the page cache **once** with a long-lived holder
([`examples/pretrain/mlock.py`](../examples/pretrain/mlock.py)), then train with
`mmap_populate=False` so reads hit the
locked cache:

```bash
# terminal 1: hold the data resident (Ctrl-C to release)
pixi run python -m examples.pretrain.mlock
# terminal 2 (same node): train with mmap_populate=False in your script
pixi run python -m examples.pretrain.phases
```

This is purely a convenience for repeated local runs; it is **not required**.
(`mlock` needs a high `RLIMIT_MEMLOCK` — e.g. `ulimit -l unlimited` or
slurm `--propagate=MEMLOCK` — to lock the full dataset.)

## Loading checkpoints

A trained run's `best_clf.safetensors` / `best_reg.safetensors` (+ the run's
`config.json`) load directly:

```python
from rt.model import load_rt_model
model, config = load_rt_model("~/ckpts/run1/best_clf.safetensors", device="cuda")
```

The same call loads a released Hub checkpoint
(`load_rt_model("stanford-star/rt-j")`). Use the resulting checkpoints for
[inference](inference.md).
