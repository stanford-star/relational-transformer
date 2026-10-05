# Preprocess

RT reads a tensor format written by the `rustler` preprocessor from any
database in RelBench format. A database is a local path or a Hub spec
`org/repo[/subdir]`, e.g. `stanford-star/relbench-v1/rel-f1`.

Preprocessing is two steps: `rustler` builds the graph (CPU, multithreaded),
then the strings are embedded (every visible GPU). The result is one
self-contained `<out_dir>/<db>/` directory.

## One database

```python
from rt.preprocess import one

one(dataset="stanford-star/relbench-v1/rel-f1", out_dir="data/relbench-preprocessed",
    embedder="all-MiniLM-L12-v2", batch_size=1024,
    skip_tasks=False, embed=True,
    upload_repo=None, public=False, revision=None)
```

`skip_tasks=True` ingests the db tables only; `embed=False` stops after
rustler; `upload_repo` pushes the result to the Hub.

## A collection

`many` takes a Hub repo of databases and processes shard `shard` of
`num_shards`, skipping finished ones:

```python
from rt.preprocess import ls, many

ls(repo="stanford-star/the-join", revision=None)
many(repo="stanford-star/the-join", out_dir="data/the-join-preprocessed",
     shard=0, num_shards=1, skip_existing=True, ...)
```

[`examples/preprocess/`](../examples/preprocess) is how the released
collections were built: per-database jobs, rustler and embed separately.

## Using the output

A `pre_dir` is always a local directory with one subdirectory per database.
Your own output and a downloaded collection ([downloads.md](downloads.md)) are
interchangeable.
