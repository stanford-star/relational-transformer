# rel2tab: tabular baselines

`rel2tab` is RT's tabular-baseline library. It converts relational prediction
tasks into tabular (train, test) pairs and runs a **featurizer + predictor**
pipeline on them, through the **same eval path** as RT
([inference.md](../../../docs/inference.md)). A baseline produces the same
RelBench submission directory as an RT eval and is scored with RelBench's own
leaderboard evaluator, so the two are directly comparable.

A baseline is a `(featurizer, predictor)` pair: each task's in-context training
labels (and optional features) are fed to a tabular predictor. The two extension
points are **featurizers** (row selection / feature extraction) and
**predictors** (train-set → prediction).

The narrow copy under [`reproduce/baselines/rel2tab`](../../../reproduce/baselines/rel2tab)
is the one the paper's numbers were produced with; this package is its
generalized successor.

## Running a baseline

```bash
pip install "relational-transformer[baselines]"
python scripts/baseline.py --featurizer entity --predictor ridge \
  --pre-dir data/relbench-preprocessed \
  --db-task-list "$(python -c 'import rt.data; print(rt.data.get_mixture_path("relbench", "forecast"))')" \
  --out-dir baseline_out
```

- **Featurizers** (`--featurizer`): `global`, `entity`, `rt` (RT embeddings —
  pass a checkpoint with `--rt-ckpt`).
- **Predictors** (`--predictor`): `mean`, `linear`, `ridge`, `xgboost`.

The `global`/`entity` featurizers with the `mean`/`linear`/`ridge` predictors
need no GPU (only the `rt` featurizer runs a model). `--out-dir` is a valid
RelBench submission directory, scored and re-validatable exactly like RT's eval
output. The context flags (`--ctx-size`, `--local-ctx-size`, `--bfs-width`, …)
match `examples/eval/plan.py` — see
[context engineering](../../../docs/inference.md#context-engineering).

The other featurizers and predictors below are not exposed by `scripts/baseline.py`;
compose them in Python via `Rel2TabModelConfig`. Feature-heavy featurizers
(`SQLFeaturizer`, `RDBLearnFeaturizer`) can be run once over every row with
`python -m rt.rel2tab.featurize` and then read back with `PrecomputedFeaturizer`.

## Architecture

```
src/rt/rel2tab/
  featurizer.py          # Featurizer ABC
  predictor.py           # Predictor ABC
  model.py               # Rel2TabModel (orchestrates the pipeline)
  config.py              # Rel2TabModelConfig, FeaturizerConfig/PredictorConfig unions
  featurize.py           # CLI: precompute a featurizer's features for every row
  featurizers/
    global_featurizer.py  # simplest example — good starting template
    entity_featurizer.py
    rt_featurizer.py
    sql_featurizer.py
    rdblearn_featurizer.py
    precomputed_featurizer.py
  predictors/
    mean_predictor.py     # simplest example — good starting template
    linear_predictor.py
    ridge_predictor.py
    xgboost_predictor.py
    lgbm_predictor.py
    tab_predictor.py
    tabicl_batched_predictor.py
    identity_predictor.py
```

## How prediction works

For each batch, `Rel2TabModel.predict` runs three steps:

1. **Extract task nodes** — finds all label cells in the batch, yielding
   per-node `labels`, `f2p_nbr_idxs` (foreign-key parent indices),
   `is_target` flags, and positional info.

2. **Featurize** — calls `featurizer.compute_features(task, node_idxs, device,
   batch_size)` once per batch to produce an (N, d_feat) feature tensor (or
   None).

3. **Predict** — for each (batch-item, context-size), filters to visible train
   rows, then calls:
   - `featurizer.featurize(train_labels, train_f2ps, target_f2p, train_feats,
     test_feat)` → returns `(filtered_train_feats, filtered_train_labels,
     filtered_test_feat)`
   - `predictor.predict(train_feats, train_labels, test_feat, task_type)` →
     returns a scalar float

## Adding a new featurizer

Create a single file `src/rt/rel2tab/featurizers/my_featurizer.py`:

```python
from dataclasses import dataclass
from rt.rel2tab.featurizer import Featurizer


@dataclass
class MyFeaturizerConfig:
    """Declare any hyperparameters as fields here."""

    some_param: int = 42

    def build(self, device):
        return MyFeaturizer(some_param=self.some_param)


class MyFeaturizer(Featurizer):
    def __init__(self, some_param):
        self.some_param = some_param

    def compute_features(self, task, node_idxs, device, batch_size):
        """Called once per batch with all N task-node indices.

        Args:
            task: Eval Task (has .db_name, .table_name, .split, .task_type, etc.)
            node_idxs: 1-D LongTensor of length N
            device: torch device
            batch_size: suggested micro-batch size

        Return an (N, d_feat) Tensor, or None if no features are produced.
        """
        return None

    def featurize(self, train_labels, train_f2ps, target_f2p, train_feats, test_feat):
        """Called per (batch-item, context-size). Filter or transform rows.

        Args:
            train_labels: 1-D float Tensor of visible train labels
            train_f2ps: (num_train, F) LongTensor — foreign-key parent indices
            target_f2p: (F,) LongTensor — target row's foreign-key parents
            train_feats: (num_train, d_feat) Tensor or None
            test_feat: (d_feat,) Tensor or None

        Return (train_feats, train_labels, test_feat) — any may be None.
        """
        return train_feats, train_labels, test_feat
```

Then register it:

1. **`featurizers/__init__.py`** — add the import:
   ```python
   from rt.rel2tab.featurizers.my_featurizer import MyFeaturizer, MyFeaturizerConfig
   ```

2. **`config.py`** — add `MyFeaturizerConfig` to the union:
   ```python
   FeaturizerConfig = (
       GlobalFeaturizerConfig
       | EntityFeaturizerConfig
       | RTFeaturizerConfig
       | MyFeaturizerConfig
   )
   ```

That's it. `Rel2TabModelConfig.build(device)` will call
`my_config.build(device)` automatically.

## Adding a new predictor

Create `src/rt/rel2tab/predictors/my_predictor.py`:

```python
from dataclasses import dataclass
from rt.rel2tab.predictor import Predictor


@dataclass
class MyPredictorConfig:
    """Declare any hyperparameters as fields here."""

    def build(self):
        return MyPredictor()


class MyPredictor(Predictor):
    def predict(self, train_features, train_labels, test_features, task_type):
        """Produce a scalar prediction for one target row.

        Args:
            train_features: (num_train, d_feat) Tensor or None
            train_labels: 1-D float Tensor (may be empty)
            test_features: (d_feat,) Tensor or None
            task_type: "clf" or "reg"

        Return a float: probability in [0,1] for clf, real value for reg.
        Convention for empty train data: 0.5 for clf, 0.0 for reg.
        """
        if len(train_labels) == 0:
            return 0.5 if task_type == "clf" else 0.0
        return train_labels.mean().item()
```

Then register it:

1. **`predictors/__init__.py`** — add the import.
2. **`config.py`** — add to the `PredictorConfig` union.

## Existing examples

| Featurizer | What it does | Config fields |
|---|---|---|
| `GlobalFeaturizer` | Passes all rows, no features | (none) |
| `EntityFeaturizer` | Filters to same-entity rows via `f2p_nbr_idxs` | (none) |
| `RTFeaturizer` | Builds local contexts, runs RT model for embeddings | RT model params, checkpoint, sampler params |
| `SQLFeaturizer` | Hand-written DuckDB feature queries per RelBench task | `pre_dir`, `db_task_list`, `splits` |
| `RDBLearnFeaturizer` | Deep-feature-synthesis features via `rdblearn` | `pre_dir`, `db_task_list`, `splits`, `max_depth`, `max_train_samples` |
| `PrecomputedFeaturizer` | Reads features written by `rt.rel2tab.featurize` | `pre_dir`, `db_task_list`, `splits`, `features_subdir` |

| Predictor | What it does | Config fields |
|---|---|---|
| `MeanPredictor` | Returns mean of train labels | (none) |
| `LinearPredictor` | Fits sklearn linear/logistic regression | (none) |
| `RidgePredictor` | Fits sklearn ridge/logistic regression with built-in CV | (none) |
| `XGBoostPredictor` | Fits gradient-boosted trees, optional hyperparameter tuning | XGBoost hyperparameters |
| `LGBMPredictor` | Fits LightGBM | `n_estimators`, `num_leaves`, `learning_rate`, `min_child_samples`, `reg_lambda` |
| `TabPredictor` | In-context TabICL or TabPFN, one item at a time | `model`, `num_workers` |
| `TabICLBatchedPredictor` | TabICL with items binned and batched per forward pass | `max_batch_size`, `min_bin_size`, `softmax_temperature`, `use_amp` |
| `IdentityPredictor` | Returns the test feature itself (for precomputed predictions) | (none) |

## Composing baselines

Featurizers and predictors compose freely:

- **Global mean** = `GlobalFeaturizerConfig()` + `MeanPredictorConfig()`
- **Entity mean** = `EntityFeaturizerConfig()` + `MeanPredictorConfig()`
- **RT + linear** = `RTFeaturizerConfig(...)` + `LinearPredictorConfig()`

## Key types to know

- **`f2p_nbr_idxs`**: Per-cell tensor of foreign-key-to-primary-key neighbor
  indices. Two rows with equal `f2p_nbr_idxs` belong to the same entity (e.g.
  same user). Shape is `(F,)` per row where F is the number of FK relations.
- **`task_type`**: Either `"clf"` (binary classification) or `"reg"`
  (regression).
- **`task`**: A namedtuple with `.db_name`, `.table_name`, `.split`,
  `.task_type`, `.target_column`, `.leakage_columns`.
