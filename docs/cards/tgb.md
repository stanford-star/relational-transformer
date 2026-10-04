---
license: cc-by-4.0
tags:
- relbench
- relational-deep-learning
- temporal-graph
pretty_name: TGB (Temporal Graph Benchmark) in RelBench format
configs:
- config_name: databases
  data_files:
  - split: eval
    path: STATS/databases.parquet
- config_name: tasks
  data_files:
  - split: eval
    path: STATS/tasks.parquet
---

# TGB datasets in RelBench format

This repository hosts the [Temporal Graph Benchmark (TGB)](https://tgb.complexdatalab.com/)
datasets exported to the [RelBench](https://github.com/snap-stanford/relbench) manifest
format. Each dataset lives in its own subdirectory with a self-describing
`manifest.yaml`, plain-parquet tables under `db/`, per-task labels under `tasks/<task>/`,
and a rendered `schema.svg`. Every task here is `kind: external`: the labels, splits, and
official negative samples are produced by TGB and served as-is.

Load any dataset by its subdirectory path:

```python
import relbench
ds = relbench.load_dataset("relbench/tgb", revision="<pin>")  # via the registry, or:
# point directly at a subdir you have locally / downloaded
task = relbench.load_task("<dataset>", "<task>")
```

## Datasets

### Dynamic link property prediction (`tgbl-*`)

Bipartite or monopartite temporal interaction networks. Task `src-dst-mrr`: predict the
next destination for a source, ranked against TGB's official negative samples
(one-vs-many MRR / Hits@k). Official `val`/`test` negatives ship under each dataset's
`negatives/` directory.

| dataset | domain |
|---|---|
| `tgbl-wiki`, `tgbl-wiki-v2` | Wikipedia editor-page edits |
| `tgbl-review`, `tgbl-review-v2` | Amazon electronics user-product reviews |
| `tgbl-coin` | cryptocurrency address transactions |
| `tgbl-comment` | Reddit reply network |
| `tgbl-flight` | global airport-flight network |

### Heterogeneous dynamic link prediction (`thgl-*`)

Temporal heterogeneous graphs with multiple node types (`nodes_type_*`) and edge types
(`events_edge_type_*`). One task per edge type (`edge-type-<k>-mrr`), evaluated against
official negatives. Global-to-local id `mappings/` and `negatives/` ship with each
dataset (required for the TGB evaluation protocol).

| dataset | domain |
|---|---|
| `thgl-software` | GitHub open-source software interactions |
| `thgl-forum` | Reddit forum (users + subreddits) |
| `thgl-github` | GitHub interactions (large) |
| `thgl-myket` | Myket Android app market interactions |

### Dynamic node property prediction (`tgbn-*`)

Node-affinity prediction over time. Task `node-label-ndcg`: predict the per-node label
distribution at the next timestamp, evaluated by NDCG@10. Labels are carried in the
`labels` / `label_events` / `label_event_items` tables.

| dataset | domain |
|---|---|
| `tgbn-trade` | UN international agriculture trade |

## Notes on evaluation

The RelBench manifest records `evaluator: tgb` on each task. TGB uses a custom protocol
(one-vs-many MRR / Hits@k for link tasks, NDCG@10 for node tasks) that differs from
RelBench's default link/classification metrics. The official negative samples and id
mappings needed to reproduce the TGB numbers are shipped alongside the data.

## Citation

If you use these datasets, please cite the TGB papers
([Huang et al., 2023](https://arxiv.org/abs/2307.01026); THGB extensions) and follow the
licensing of each original data source. See <https://tgb.complexdatalab.com/>.
