# Examples

Everything here runs on RelBench or on your own data in the same format. The
recipes behind the released checkpoints and leaderboard entries live beside the
notes on how the pieces work. For a first look, start with the
[notebooks](../notebooks/).

| | what | cost |
|---|---|---|
| [`preprocess/`](preprocess/) | RelBench-format databases -> RT's tensor format | CPU per database, one GPU to embed |
| [`pretrain/`](pretrain/) | PluRel, then the Join warm-started from it | 41 B200-h (~100 A100-h) to the released steps |
| [`eval/`](eval/) | one checkpoint, one context, every test split | ~5 A100-min per task |
| [`icl/`](icl/) | leaderboard entries: RT-J and RT-PluRel in-context | ~9.5 A100-h per task |
| [`finetune/`](finetune/) | leaderboard entries: RT, RT-J and RT-PluRel fine-tuned | 6.4-8.6 A100-h per task |

## How a recipe runs

There is no CLI and no scheduler. Each recipe builds a list of `Job`s, each a
call to a function in `rt` with every argument spelled out, and
[`launch.py`](launch.py) runs them in the current process. Hand the list to
your own scheduler if you have one. Finished stages are skipped, so re-running
a plan resumes it. Run from the repo root:

```bash
pixi run python -m examples.<recipe>.<stage>
```

## Data

Raw data, preprocessed data and checkpoints are on HuggingFace under
[`stanford-star`](https://huggingface.co/stanford-star). Checkpoints are
fetched on demand. Data is downloaded up front to the `data/` paths the
recipes expect; a `pre_dir` is always a local directory with one subdirectory
per database.

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed       # eval / validation
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed       # pretraining
pixi run python -m examples.preprocess.task_lists   # -> data/db-task-lists/*.json
```

A subset is a database (`--include "rel-f1/*"`), or the core files plus one
embedder:

```bash
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed --max-workers 16 \
  --include "*/meta.json" "*/table_info.json" "*/column_index.json" \
            "*/nodes.rkyv" "*/offsets.rkyv" "*/p2f_adj.rkyv" \
            "*/text_emb_all-MiniLM-L12-v2.bin"
```

The repos are rewritten in place. The released numbers come from these
revisions; pass `--revision <sha>` to reproduce them.

| repository | revision | date |
|---|---|---|
| `stanford-star/the-join-preprocessed` | `ba5574ba659f0ef8b592793ed2ac9cb78dc87100` | 2026-09-12 |
| `stanford-star/relbench-preprocessed` | `1016626ddb30c027b92458bf866903850cc205e1` | 2026-08-08 |
| `stanford-star/plurel-preprocessed` | `9d70172425b44053f1270b19094ba6dcf7d66464` | 2026-09-12 |
| `stanford-star/the-join` | `ec028dddca63c7eeb49bbce3d7a713783e165eea` | 2026-08-04 |
| `stanford-star/relbench-v1` | `d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600` | 2026-08-25 |
| `stanford-star/plurel` | `ae2f0b04f71ec17aebf6cf1fa8259fa655994b58` | 2026-09-08 |
| `stanford-star/relbench-raw` | `f1d7228af23a22b9ece756fe5dda4dda79b711dc` | 2026-08-25 |
| `stanford-star/rt-j` | `360798f5335975fdcae73e3ed58cf03366dbc082` | 2026-09-14 |
| `stanford-star/rt-plurel` | `d27c97b045fc4f504848f15c730acb87970aac1d` | 2026-09-11 |

## Tools

[`ctx_viz.py`](ctx_viz.py) is a web UI over the contexts the sampler builds:
pick a database, task and row, set the sampler knobs, and see what the model
attends over.

```bash
pixi run python examples/ctx_viz.py --pre-root data/relbench-preprocessed
```
