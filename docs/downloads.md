# Downloads

Raw data, preprocessed data, and checkpoints live on HuggingFace under
[`stanford-star`](https://huggingface.co/stanford-star). Download data up front
with the `hf` CLI; a `pre_dir` is always a local directory. Checkpoints are
fetched on demand (`load_rt_model("stanford-star/rt-j")`,
`--model.load-ckpt-path stanford-star/rt-j`).

```bash
# Preprocessed "the Join" -- pretraining data
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed

# Preprocessed RelBench -- validation/eval data
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed

# Raw data, only to re-run preprocessing (see preprocess.md)
pixi run hf download stanford-star/the-join --repo-type dataset
pixi run hf download stanford-star/relbench-v1 --repo-type dataset

# Checkpoint
pixi run hf download stanford-star/rt-j --repo-type model
```

The `data/*-preprocessed` paths are the scripts' defaults
(`--train.pre-dir`, `--eval.pre-dir`). The `(db, task)` lists a run trains or
evaluates on are generated from the downloaded data, not shipped:
`pixi run python -m examples.preprocess.task_lists` writes `data/db-task-lists/{rt-j,rt-plurel-train,relbench-forecast}.json`
(the PluRel list also needs the raw `stanford-star/plurel` repo's `*/manifest.yaml`
and `*/scores.json`).

To fetch a subset, keep the core rustler artifacts plus the one text embedder
you train with, and/or restrict to databases (`--include "<db>/*"`):

```bash
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed --max-workers 16 \
  --include "*/meta.json" "*/table_info.json" "*/column_index.json" \
            "*/nodes.rkyv" "*/offsets.rkyv" "*/p2f_adj.rkyv" \
            "*/text_emb_all-MiniLM-L12-v2.bin"
```

## Revisions

The repos are rewritten in place. The RT-J paper's numbers were produced
against these revisions; pass `--revision <sha>` to reproduce them.

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

The checkpoint and the `rustler` commit (recorded in each card) fix a result too.
