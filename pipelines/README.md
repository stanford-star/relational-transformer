# Pipelines

The four recipes behind the released checkpoints and the leaderboard entries.
Each runs on RelBench or on your own data in the same format.

| | what | cost |
|---|---|---|
| [`preprocess`](preprocess/) | RelBench-format databases -> RT's tensor format | CPU per database, then one GPU to embed |
| [`pretrain`](pretrain/) | two phases: PluRel, then the Join warm-started from it | thousands of GPU-h |
| [`icl`](icl/) | frozen checkpoint: context search, then a top-4 x 4-seed ensemble | ~200 GPU-h |
| [`finetune`](finetune/) | per-task: select, context search, refit, 8-seed ensemble | ~135 GPU-h |

Paper figures and tables are [`../reproduce/`](../reproduce/). For a first look
at your own database without a pipeline, use [`../byod/`](../byod/).

No scheduler ships here. A pipeline enumerates jobs and
[`../reproduce/launch.py`](../reproduce/launch.py) runs them in the current
process; `run(jobs, launcher)` takes your own. Stages skip completed work, so
re-running a plan is how you resume.

Pin the revisions in [`../docs/downloads.md`](../docs/downloads.md); expect
agreement to a few tenths of a point.
