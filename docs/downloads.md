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
(`--train.pre-dir`, `--eval.pre-dir`). The `(db, task)` mixtures are vendored
in the package: `rt.data.list_mixtures()`, `rt.data.get_mixture(collection, name)`.

To fetch a subset, keep the core rustler artifacts plus the one text embedder
you train with, and/or restrict to databases (`--include "<db>/*"`):

```bash
pixi run hf download stanford-star/the-join-preprocessed --repo-type dataset \
  --local-dir data/the-join-preprocessed --max-workers 16 \
  --include "*/meta.json" "*/table_info.json" "*/column_index.json" \
            "*/nodes.rkyv" "*/offsets.rkyv" "*/p2f_adj.rkyv" \
            "*/text_emb_all-MiniLM-L12-v2.bin"
```
