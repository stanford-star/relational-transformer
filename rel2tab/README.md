# rel2tab

rel2tab converts relational prediction tasks into tabular (train, test) pairs
and runs a featurizer + predictor pipeline on them. The two extension points
are **featurizers** (row selection / feature extraction) and **predictors**
(train-set → prediction).

## Architecture

```
rel2tab/
  featurizer.py          # Featurizer ABC
  predictor.py           # Predictor ABC
  model.py               # Rel2TabModel (orchestrates the pipeline)
  config.py              # Rel2TabModelConfig, FeaturizerConfig/PredictorConfig unions
  featurizers/
    global_featurizer.py  # simplest example — good starting template
    entity_featurizer.py
    rt_featurizer.py
  predictors/
    mean_predictor.py     # simplest example — good starting template
    linear_predictor.py
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

Create a single file `rel2tab/featurizers/my_featurizer.py`:

```python
from dataclasses import dataclass
from rel2tab.featurizer import Featurizer


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

1. **`rel2tab/featurizers/__init__.py`** — add the import:
   ```python
   from rel2tab.featurizers.my_featurizer import MyFeaturizer, MyFeaturizerConfig
   ```

2. **`rel2tab/config.py`** — add `MyFeaturizerConfig` to the union:
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

Create `rel2tab/predictors/my_predictor.py`:

```python
from dataclasses import dataclass
from rel2tab.predictor import Predictor


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

1. **`rel2tab/predictors/__init__.py`** — add the import.
2. **`rel2tab/config.py`** — add to the `PredictorConfig` union.

## Existing examples

| Featurizer | What it does | Config fields |
|---|---|---|
| `GlobalFeaturizer` | Passes all rows, no features | (none) |
| `EntityFeaturizer` | Filters to same-entity rows via `f2p_nbr_idxs` | (none) |
| `RTFeaturizer` | Builds local contexts, runs RT model for embeddings | RT model params, checkpoint, sampler params |

| Predictor | What it does | Config fields |
|---|---|---|
| `MeanPredictor` | Returns mean of train labels | (none) |
| `LinearPredictor` | Fits sklearn linear/logistic regression | (none) |
| `RidgePredictor` | Fits sklearn ridge/logistic regression with built-in CV | (none) |
| `XGBoostPredictor` | Fits gradient-boosted trees, optional hyperparameter tuning | XGBoost hyperparameters |

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
