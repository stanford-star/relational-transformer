# Per-task fine-tuning

One checkpoint fine-tuned per RelBench task. Four stages: selection arm on
`train`, context search against the fine-tuned weights, refit on `train + val`,
8-seed test ensemble. One GPU per job, 21 jobs per model.

## Reproduce ours

`RT_PRE_DIR` (preprocessed RelBench data) and `RT_OUT_ROOT` (writable) must be
set; both fail loudly if missing.

```bash
pixi run python -m examples.finetune.plan    rt-j      # or rt-plurel, rt
pixi run python -m examples.finetune.collect rt-j      # scores, writes the zips
```

Three models, identical stages, differing only in what they start from
([`plan.py`](plan.py) `MODELS`):

| model | warm start | entry | clf roc_auc | reg nmae | cost |
|---|---|---|--:|--:|--:|
| `rt-j` | [`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j) | RT-J (fine-tuned) | **0.7902** | **0.2711** | 135 GPU-h |
| `rt-plurel` | [`stanford-star/rt-plurel`](https://huggingface.co/stanford-star/rt-plurel) | RT-PluRel (fine-tuned) | 0.7853 | 0.2757 | 181 GPU-h |
| `rt` | none — random init | RT (from scratch) | 0.7776 | 0.2927 | — |

With no warm start the arms train the whole model instead of a zero-initialised
additive delta (`delta_finetune`). Per task: 2.7–3.7 h on rel-f1, 3–7 h on
most, 10–21 h where the selection arm hits its 50k-step ceiling.

Outputs land under `$RT_OUT_ROOT/no-wandb-entity/finetune/<model>-<db>-<task>-{inner,tune,outer,test}/`,
with each task's prediction table in the `-test/eval_out/` directory. Stages are
idempotent: a rerun resumes from `resume.pt` and skips finished stages.

`collect` was verified end to end and reproduces every number above. The stage
bodies were not re-run on a GPU from this tree.

## Run it on your own data

Change `RT_PRE_DIR`, the task list in [`plan.py`](plan.py), and the warm start.
Stages 1–3 work unconditionally; stage 4 builds its prediction table through
`relbench.load_dataset(meta["source"])`, so without a relbench-loadable
`source` stop after `outer` and score the refit yourself.
