# Preprocessing a collection

Whole collections into RT's tensor format. Two jobs per database: `rustler`
(CPU, builds the graph) then `embed` (one GPU, embeds the strings).

## Reproduce ours

```bash
pixi run hf download stanford-star/the-join --repo-type dataset \
  --revision ec028dddca63c7eeb49bbce3d7a713783e165eea --local-dir data/the-join
pixi run hf download stanford-star/plurel --repo-type dataset \
  --revision ae2f0b04f71ec17aebf6cf1fa8259fa655994b58 --local-dir data/plurel
pixi run hf download stanford-star/relbench-v1 --repo-type dataset \
  --revision d8e976fd0a4b78877204bc8dfbcfc9a9f7f48600 --local-dir data/relbench-v1

pixi run python -m examples.preprocess.plan      # every outstanding job, in this process
pixi run python -m examples.preprocess.task_lists  # -> data/db-task-lists/*.json
```

Per database it writes:

```
<db>/  meta.json  table_info.json  column_index.json
       nodes.rkyv  offsets.rkyv  p2f_adj.rkyv
       text.json  text_emb_all-MiniLM-L12-v2.bin
```

Already-built databases are not planned again, so re-running the plan resumes
it. `rustler` scales with output size, `embed` with text volume — the two
`rel-amazon` databases carry 10.5 GiB of text each and dominate any RelBench
run. Both stages were verified end to end on `rel-f1`.

## Run it on your own data

Input must be RelBench v3 format:

```
<db>/
  manifest.yaml          # the sole source of relational metadata
  db/<table>.parquet     # one file per table, native dtypes only
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
