# Pipelines

The four pipelines behind the released checkpoints and the RelBench leaderboard
entries, each runnable on RelBench or on your own data in the same format.

| Pipeline | What it does | Needs |
|---|---|---|
| [`preprocess`](preprocess/) | A collection of RelBench-format databases into RT's on-disk tensor format | CPU per database, then one GPU to embed |
| [`pretrain`](pretrain/) | The two-phase recipe behind `stanford-star/rt-j`: PluRel, then the Join warm-started from it | thousands of GPU-hours |
| [`icl`](icl/) | In-context prediction with a frozen checkpoint: per-task context search, then a top-4 x 4-seed ensemble | ~200 GPU-hours per checkpoint |
| [`finetune`](finetune/) | Per-task fine-tuning: selection arm, context search, refit, 8-seed test ensemble | ~135 GPU-hours per checkpoint |

Each README has the same two sections: **Reproduce ours**, with the commands,
the expected numbers and the measured cost, and **Run it on your own data**,
with the short list of things that change.

## Which one you want

- **Predict on your own database with a released checkpoint**, no training:
  [`preprocess`](preprocess/) then [`icl`](icl/). For a single database and a
  quick look, [`byod/`](../byod/) is faster — a notebook, no pipeline.
- **Get the most out of a released checkpoint on your own tasks**:
  [`preprocess`](preprocess/) then [`finetune`](finetune/).
- **Reproduce a leaderboard entry**: [`icl`](icl/) for the in-context entries,
  [`finetune`](finetune/) for the fine-tuned ones. The preprocessed RelBench
  data is published, so neither needs `preprocess`.
- **Reproduce the paper's figures and tables**: [`../reproduce/`](../reproduce/),
  which carries the committed per-figure series as well as the code.
- **Train a checkpoint of your own**: [`pretrain`](pretrain/).

## Running the work

None of these ship a scheduler. A pipeline enumerates its jobs and
[`reproduce/launch.py`](../reproduce/launch.py) runs them one at a time in the
current process; `run(jobs, launcher)` takes your own launcher if you have a
cluster. Every stage is idempotent per unit of work: a stage whose output
exists is not planned again, so re-running a plan is how you resume it.

Pin the dataset revisions recorded in [`docs/downloads.md`](../docs/downloads.md)
and expect agreement with the published numbers to a few tenths of a point.
