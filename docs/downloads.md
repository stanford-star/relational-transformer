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

