# Pretraining RT-J

The two-phase pretraining recipe behind
[`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j), with every
argument spelled out at the call site:
[`phases.py`](phases.py). Phase 1 pretrains on the synthetic PluRel corpus;
phase 2 continues on the Join, **warm-started from the phase-1 weights**. The
released checkpoint comes from phase 2.

[`examples/train.py`](../../examples/train.py) is the minimal single-run example
— one call, one phase, edit and go. This directory is the full recipe: both
phases, in order, with the data each reads and the checkpoint each produces.

The library reference for every knob is [`docs/train.md`](../../docs/train.md).

## Reproduce ours

### 0. Data

Three preprocessed collections, all local directories, nothing fetched on
demand. Pin the revisions recorded in
[`docs/downloads.md`](../../docs/downloads.md) — an unpinned download is not the
data this checkpoint was trained on.

```bash
pixi run hf download stanford-star/plurel-preprocessed --repo-type dataset \
  --revision 9d70172425b44053f1270b19094ba6dcf7d66464 \
  --local-dir data/plurel-preprocessed
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --revision ba5574ba659f0ef8b592793ed2ac9cb78dc87100 \
  --local-dir data/the-join-preprocessed
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --revision 1016626ddb30c027b92458bf866903850cc205e1 \
  --local-dir data/relbench-preprocessed
```

Or build them yourself with [`../preprocess`](../preprocess).

| | phase 1 | phase 2 | validation (both phases) |
|---|---|---|---|
| `pre_dir` | `plurel-preprocessed` (~40 GiB) | `the-join-preprocessed` (~256 GiB) | `relbench-preprocessed` |
| mixture | `plurel/rt-plurel-train` | `the-join/rt-j` | `relbench/forecast` |
| pairs | 86,211 tasks over 1,900 synthetic databases | 13,243 tasks over the 523 real databases under the 5 GB per-database cutoff | 21 RelBench forecast tasks, `val` split |

The mixtures are vendored in the package and read with
`rt.data.get_mixture_path(collection, name)`; `phases.py` resolves all three, so
there is no list to download or keep in sync.

### 1. The two phases, in order

```bash
# both phases, sequentially, in this process
CUDA_VISIBLE_DEVICES=0,1 pixi run python -m pipelines.pretrain.phases

# the job list without running anything
pixi run python -c "from pipelines.pretrain.phases import rt_j_jobs; from reproduce.launch import describe; describe(rt_j_jobs())"
```

The two phases are **identical calls except for three arguments** — this is the
whole difference, and `phases.py` is written so you can see that:

| | phase 1 | phase 2 |
|---|---|---|
| `load_ckpt_path` | `None` (from scratch) | `"stanford-star/rt-plurel"` (the phase-1 weights) |
| `db_task_list` | `plurel/rt-plurel-train` | `the-join/rt-j` |
| `pre_dir` | PluRel | the Join |

Everything else is shared: 12 blocks, `d_model` 512, 8 heads, `d_ff` 2048
(85,562,530 parameters), `all-MiniLM-L12-v2` text embeddings at `d_text` 384,
Huber loss, Muon for the hidden weight matrices and AdamW for the rest,
constant `lr=5e-4` after a 2,000-step linear warmup and no decay, `wd=0.1`,
gradient clipping at global norm 1.0, `total_bs=1024`, SWA EMA at
`swa_momentum=0.9995`, `mask_prob_max=0.0` (only the target cell is masked),
`seed=0`, `total_steps=2**15 + 1` with `early_stop_after_steps=10_000`, and
validation every 1,000 steps on the 21 RelBench forecast tasks' `val` splits at
`ctx_size=8192`, `(local_ctx_size, bfs_width, prefer_latest) = (256, 32, True)`.

Context shapes are resampled per batch from `ctx_size ∈ {512, 1024, 2048, 4096,
8192}`, `local_ctx_size ∈ {128, 256, 512, 1024, 2048, 4096, 8192}`,
`bfs_width ∈ {8, 16, 32, 64, 128, 256}`, `prefer_latest ∈ {False, True}`, over
10,000 random walks of length 20.

Phase 2 loads `stanford-star/rt-plurel` — the released phase-1 checkpoint —
rather than the phase-1 run's own output directory, which is exactly what the
released run did and is what makes **phase 2 runnable on its own**. To chain
your own phase 1 instead, pass its selected checkpoint:

```python
phase_two_load_ckpt_path="~/ckpts/rt-j-pretrain/<phase-1 run_id>/best_live_reg.safetensors"
```

A run writes its checkpoints under `<out_root>/<project>/<run_id>/` and resumes
automatically from `resume.pt` in that directory if relaunched with the same
`run_id`, at any GPU count.

### 2. Which step the release is

Neither phase stops at its selected step: the run keeps training and its own
in-loop selection writes the best-so-far checkpoints, `best_*.safetensors`. The
released weights are those files.

| phase | released as | file the run wrote | step | selected by |
|---|---|---|---|---|
| 1 | [`stanford-star/rt-plurel`](https://huggingface.co/stanford-star/rt-plurel) | `best_live_reg.safetensors` (the live net) | 8,000 | mean validation nMAE |
| 2 | [`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j) | `best_swa_clf.safetensors` (the SWA/EMA net) | 9,000 | mean validation AUROC, **74.36** |

Selection is on the **validation** split only; the classification-selected
checkpoint is the single released one and serves regression too. `rt-j`'s
`model.safetensors` records `step=9000`, `swa_n=9000` in its metadata, which is
how you check a checkpoint is this one.

Note that 9,000 and 8,000 are far short of `total_steps`: a reader who trains to
`total_steps` and takes the last weights has not reproduced either checkpoint.
Take `best_swa_clf.safetensors`.

### 3. Measured cost

| | |
|---|---|
| Shape per phase | 2 GPUs, one node. `tokens_per_gpu` was `2**18` on the B200s used; `2**17` fits an 80 GB A100 and `2**16` an H100. `total_bs=1024` fixes the global batch, so `tokens_per_gpu` changes throughput and not the data a step sees. |
| Time to first step | 25–45 min on a node that has not read the data recently, ~4 min on one that has (page-cache population; `compile=True` is about 1 min of it). |
| Total | thousands of GPU-hours for both phases. We have no single wall-clock figure to quote: both phases ran preemptibly across requeues, resuming from `resume.pt`. |
| RAM | give the process as much of the node as you can. `mmap_populate=True` pulls the mixture into the page cache at startup; on a shared filesystem that evicts it, pass `stage_dir` to copy onto node-local storage first. |

Multi-GPU is automatic under any launcher that sets `RANK`, `LOCAL_RANK` and
`WORLD_SIZE` per process, one process per GPU
(`torchrun --standalone --nproc-per-node=auto -m pipelines.pretrain.phases`).
A GPU is required: `flex_attention` has no fused bfloat16 CPU kernel.

### What agreement to expect

Context sampling is stochastic, so a fresh run does not land on the published
numbers digit-for-digit. Pin the dataset revisions recorded in
[`docs/downloads.md`](../../docs/downloads.md) and expect agreement to a few
tenths of a point.

### What has actually been run

- **Verified.** `rt_j_jobs()` enumerates the two phases; both job argument sets
  bind against `rt.train.main`'s signature with no missing and no extra
  argument, and the two differ in exactly `load_ckpt_path`, `db_task_list`,
  `pre_dir` (plus `run_id` and `run_name`). The mixtures resolve, and their
  sizes are the published ones: 86,211 pairs over 1,900 databases for phase 1,
  13,243 over 523 for phase 2, 21 tasks for validation.
- **Not verified here.** The runs themselves. Pretraining is thousands of
  GPU-hours and is not re-run to check a recipe; the arguments above are the
  ones the released checkpoints were produced with.

## Run it on your own data

`phases.py` separates the three things worth changing — the mixture, the data,
and the budget — from the ~50 arguments that are the recipe. `jobs()` takes
every one of them explicitly and defaults nothing; `rt_j_jobs()` is the released
set of values and is what you edit or replace.

### Your own task mixture

`db_task_list` is `(db, task)` pairs: a list of tuples, or a path to a JSON file
of pairs. Names resolve against the tasks each database ships in its
`meta.json`. Either pass a vendored mixture by name,

```python
from rt.data import get_mixture_path, list_mixtures

list_mixtures()                                  # every (collection, name)
get_mixture_path("the-join", "forecast")         # 4,098 forecast-only pairs
```

or your own file, which is the usual case for your own databases:

```python
import json
json.dump([["my-db", "my-task"], ["my-db", "other-task"]], open("mixture.json", "w"))
```

and pass `"mixture.json"` as `db_task_list`. A pair naming a task the build
cannot predict is reported and ignored rather than fatal, so check the startup
log for `tasks_skipped` if your count looks wrong. `pre_dir` must be a
preprocessed directory holding every database the mixture names — see
[`../preprocess`](../preprocess) to build one from your own databases.

Keep `eval_db_task_list` and `eval_pre_dir` pointed at something you are *not*
training on; they are what selects the checkpoint. Validation on tasks in the
training mixture selects nothing meaningful.

### A smaller budget

In rough order of what buys the most per unit of pain:

- **Start from `stanford-star/rt-plurel` and run phase 2 only.** Phase 1 is
  roughly where the compute went and its product is already published. Drop
  `phase_one(...)` from `jobs()`.
- **Or start from `stanford-star/rt-j` itself** as `load_ckpt_path` and
  continue on your mixture. Set `can_select_init_model=False` (as the recipe
  does) so eval noise cannot hand the win back to the weights you began with.
- **Cut `total_steps`.** Both released checkpoints were selected within the
  first 9,000 steps of their phase, so a budget far below `2**15 + 1` is not
  obviously wrong. Leave `lr_decay_steps=0`: the decay is measured back from
  `total_steps`, so lowering `total_steps` with a decay set changes the
  schedule and not just its length.
- **Lower `eval_freq`'s cost, not its frequency.** Each eval is a full pass over
  `eval_items_per_task` items per task; 1,024 over 21 tasks is the released
  setting.
- **Narrow the context lists.** `ctx_size_list=[8192]` alone trains one context
  shape and is much cheaper per step; the released lists are what make the model
  usable across context budgets at inference.
- **`tokens_per_gpu` to fit your card**, as above — it costs time, not accuracy.
- **`swa_momentum=None`** turns SWA off: no second net, no `best_swa_*` files.
  The released RT-J *is* an SWA checkpoint, so this changes what you produce.
