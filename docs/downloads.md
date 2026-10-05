# Downloads

Everything lives on HuggingFace under
[`stanford-star`](https://huggingface.co/stanford-star). Download data up
front; checkpoints are fetched on demand (`load_rt_model("stanford-star/rt-j")`).

```bash
pixi run hf download stanford-star/relbench-preprocessed --repo-type dataset \
  --local-dir data/relbench-preprocessed       # eval / validation data
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed       # pretraining data
```

The `data/*-preprocessed` paths are what the examples expect. Task lists are
generated from the downloaded data:

```bash
pixi run python -m examples.preprocess.task_lists   # -> data/db-task-lists/*.json
```

To fetch a subset, restrict to databases (`--include "rel-f1/*"`) or to the
core files plus one embedder:

```bash
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed --max-workers 16 \
  --include "*/meta.json" "*/table_info.json" "*/column_index.json" \
            "*/nodes.rkyv" "*/offsets.rkyv" "*/p2f_adj.rkyv" \
            "*/text_emb_all-MiniLM-L12-v2.bin"
```

## Revisions

The repos are rewritten in place. The RT-J paper's numbers come from these
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
