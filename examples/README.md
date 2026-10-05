# Examples

The recipes behind the released checkpoints and leaderboard entries. Each runs
on RelBench or on your own data in the same format.

| | what | cost |
|---|---|---|
| [`preprocess`](preprocess/) | RelBench-format databases -> RT's tensor format | CPU per database, one GPU to embed |
| [`pretrain`](pretrain/) | PluRel, then the Join warm-started from it | thousands of GPU-h |
| [`eval`](eval/) | one checkpoint, one context, every test split | a few GPU-h |
| [`icl`](icl/) | frozen checkpoint: context search, then ensemble | ~200 GPU-h |
| [`finetune`](finetune/) | per-task fine-tuning and ensemble | ~135 GPU-h |

For a first look at your own database, use [`../byod/`](../byod/).

Each example builds a list of `Job`s and [`launch.py`](launch.py) runs them in
the current process; hand the list to your own scheduler if you have one.
Finished stages are skipped, so re-running a plan resumes it. Run everything
from the repo root: `pixi run python -m examples.<name>.<stage>`.

[`ctx_viz.py`](ctx_viz.py) is a web UI over the contexts the sampler builds.
