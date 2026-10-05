# Reproducing the RT-J paper

The reduced data behind every figure and table is committed in
[`series/`](series), so the figures are checkable without rerunning anything.
Each stage's `reduce.py` / `collect.py` regenerates its series from finished
runs; each `plan.py` enumerates the runs themselves.

| paper artifact | stage |
|---|---|
| main and per-task baseline figures | [`scaling/`](scaling) — `fulltest` arms |
| extended-context appendix | [`scaling/`](scaling) — `subsampled` arms |
| retriever and schema-semantics ablations | [`scaling/`](scaling) — `abl` arms |
| test-time-compute ensembling | [`enscurve/`](enscurve) |
| default-vs-tuned context table | [`valtest/`](valtest) |
| tuned+ensembled per-task table | [`leaderboard/`](leaderboard) |
| the per-task context grid feeding those three | [`tune/`](tune) |
| baseline features and FAISS indices | [`baselines/`](baselines) |
| pretraining-ablation figures | [`pretrain_abl/`](pretrain_abl) — series only |

`tune/` is ~200 GPU-hours and `baselines/` is the bulk of the rest; both of
their outputs are committed, so start downstream of them.

## Running a stage

```bash
python -m reproduce.enscurve.plan     # 42 jobs, one GPU, sequential
python -m reproduce.enscurve.reduce   # -> series/enscurve/{default,tuned}.json
```

`describe(jobs())` prints the plan without running it, and `run(jobs(), launcher)`
takes your own launcher. A finished plan reports `0 jobs`, so re-running it is
how you resume.

Order: `baselines` -> `scaling`; `tune` -> `enscurve` -> `valtest`, and
`tune` -> `leaderboard`.

| variable | what it points at |
|---|---|
| `RT_CKPT` | a local [`stanford-star/rt-j`](https://huggingface.co/stanford-star/rt-j) mirror |
| `RT_PRE_DIR` | [`relbench-preprocessed`](https://huggingface.co/datasets/stanford-star/relbench-preprocessed) |
| `RT_RAW_DIR` | [`relbench-v1`](https://huggingface.co/datasets/stanford-star/relbench-v1), baseline featurizers only |
| `RT_SHARE` | writable: baseline features, FAISS indices, TabICL checkpoints |
| `RT_OUT_ROOT` | writable: per-task result JSONs |

Pin the revisions in [`../docs/downloads.md`](../docs/downloads.md). Context
sampling is stochastic; expect a few tenths of a point, and report anything
larger.

Not reproducible here: the pretraining ablations (five multi-day runs whose
checkpoints are unreleased) and six appendix schematics that never had a
generating script. The paper's own plotting scripts read a private wandb;
[`series/`](series) is the same data, committed instead.

Questions: open an issue on
[the repo](https://github.com/stanford-star/relational-transformer/issues).
