# Preprocess

RT trains and predicts on a custom on-disk format produced by the `rustler`
preprocessor from **any dataset in relbench format**. A dataset is a local path, or a HuggingFace Hub spec `org/repo[/subdir]`
(e.g. `stanford-star/relbench/rel-f1`).

Preprocessing runs `download/resolve → rustler → text embeddings` and writes a
self-contained `<out-dir>/<name>/` directory. Text embeddings use all visible
GPUs automatically (Sentence-Transformers multi-process); rustler itself is
multithreaded (rayon).

## Preprocess one database in RelBench format

There is no CLI. Copy [`examples/preprocess.py`](../examples/preprocess.py),
edit the call, run it:

```bash
pixi run python examples/preprocess.py
```

As written it calls `one(dataset="stanford-star/relbench/rel-f1",
out_dir="data/relbench-preprocessed", ...)` and writes
`data/relbench-preprocessed/rel-f1/`, the rustler artifacts the RT dataloaders
read.

Any dataset in relbench format works by swapping the `dataset` argument — the
manifest is the sole source of relational metadata; the parquet files carry only
native dtypes. Other arguments worth knowing: `skip_tasks=True` (ingest db
tables only), `embed=False`, `embedder`, `batch_size`, and `upload_repo="<hub
repo>"` (preprocess and push in one step).

## Preprocess many databases efficiently

To preprocess a whole Hub collection (e.g. the 650-database [the Join](https://huggingface.co/datasets/stanford-star/the-join)),
call `many` instead of `one` — `preprocess_a_collection()` in the same example:

```python
ls(repo="stanford-star/the-join", revision=None)      # what is in the collection
many(repo="stanford-star/the-join", out_dir="data/the-join-preprocessed",
     shard=0, num_shards=1, skip_existing=True, ...)
```

`skip_existing=True` makes the pass resumable (datasets whose embeddings are
already written are skipped). `shard=i, num_shards=N` splits the collection
across a job array (e.g. a preemptible slurm array with `--array=0-63` mapping
the task id to `shard`).

## Using preprocessed data

Everywhere a `pre_dir` is taken (see [inference](inference.md) and
[pretrain](train.md)) it is a **local directory**: what this preprocessor wrote,
or a published collection you downloaded with `hf download --local-dir` (see
[downloads.md](downloads.md)). Nothing is fetched on demand, so you never have
to upload anything to use your own data, and a run's data is a path you can
inspect. The layout is the same either way — one subdirectory per database — so
your own output and a downloaded collection are interchangeable.
