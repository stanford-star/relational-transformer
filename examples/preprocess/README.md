# Preprocess

RT reads a tensor format written by the `rustler` preprocessor from any
database in RelBench format. Two jobs per database: `rustler` builds the
graph (CPU, multithreaded), then `embed` embeds the strings (one GPU).

## Reproduce ours

```bash
pixi run hf download stanford-star/the-join --repo-type dataset \
  --revision ec028dddca63c7eeb49bbce3d7a713783e165eea --local-dir data/the-join
pixi run hf download stanford-star/plurel --repo-type dataset \
  --revision ae2f0b04f71ec17aebf6cf1fa8259fa655994b58 --local-dir data/plurel
pixi run hf download stanford-star/relbench-v1 --repo-type dataset \
  --revision d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600 --local-dir data/relbench-v1

pixi run python -m examples.preprocess.plan         # every outstanding job
pixi run python -m examples.preprocess.task_lists   # -> data/db-task-lists/*.json
```

Only databases whose preprocessed output is under 5 GB are uploaded to
`the-join-preprocessed` and used for pretraining. Raw size does not predict
preprocessed size, so `plan` preprocesses every database and then
`drop_oversized` deletes any output directory over `MAX_PRE_BYTES`. That is
why `the-join-preprocessed` holds 523 of the 639 `the-join` databases: 67
preprocess to more than 5 GB (`nodes.rkyv` dominates; `join-se-electronics`
reaches 87 GB from 80 MB of parquet) and the other 49 ship no
binary-classification or regression task, so they yield no tasks.

Per database it writes:

```
<db>/  meta.json  table_info.json  column_index.json
       nodes.rkyv  offsets.rkyv  p2f_adj.rkyv
       text.json  text_emb_all-MiniLM-L12-v2.bin
```

`embed` scales with text volume; the two `rel-amazon` databases dominate any
RelBench run.

## Run it on your own data

Input is RelBench v3 format, local or on the Hub as `org/repo[/subdir]`:

```
<db>/
  manifest.yaml
  db/<table>.parquet
  tasks/<task>/manifest.yaml
  tasks/<task>/<split>.parquet
```

```python
from examples.preprocess.plan import jobs
from examples.launch import run_sequential

run_sequential(jobs(
    raw_dir="data/my-collection",
    out_dir="data/my-collection-preprocessed",
    source_repo="my-org/my-collection",
    embedder="all-MiniLM-L12-v2",
    batch_size=1024,
))
```

## The API underneath

`rt.preprocess.one` does one database in one call:

```python
from rt.preprocess import one

one(dataset="stanford-star/relbench-v1/rel-f1", out_dir="data/relbench-preprocessed",
    embedder="all-MiniLM-L12-v2", batch_size=1024,
    skip_tasks=False, embed=True,
    upload_repo=None, public=False, revision=None)
```

`skip_tasks=True` ingests the db tables only; `embed=False` stops after
rustler; `upload_repo` pushes the result to the Hub. `rt.preprocess.many`
takes a Hub repo of databases and does shard `shard` of `num_shards`, skipping
finished ones; `rt.preprocess.ls` lists the repo.
