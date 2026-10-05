"""LightGBM predictor for the rel2tab pipeline.

Fits a LightGBM model per prediction.  Heavily regularized by default
(few leaves, high min_child_samples) since training sets are small
(2-50 rows per prediction at typical context sizes).
"""

from dataclasses import dataclass

import numpy as np

from rt.rel2tab.predictor import Predictor


@dataclass
class LGBMPredictorConfig:
    n_estimators: int
    num_leaves: int
    learning_rate: float
    min_child_samples: int
    reg_lambda: float

    def build(self):
        return LGBMPredictor(
            n_estimators=self.n_estimators,
            num_leaves=self.num_leaves,
            learning_rate=self.learning_rate,
            min_child_samples=self.min_child_samples,
            reg_lambda=self.reg_lambda,
        )


class LGBMPredictor(Predictor):
    """Fit a LightGBM model per prediction."""

    def __init__(
        self,
        n_estimators,
        num_leaves,
        learning_rate,
        min_child_samples,
        reg_lambda,
    ):
        self.params = {
            "n_estimators": n_estimators,
            "num_leaves": num_leaves,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "reg_lambda": reg_lambda,
            "verbose": -1,
        }

    def predict(self, train_features, train_labels, test_features, task_type):
        from lightgbm import LGBMClassifier, LGBMRegressor

        if train_features is None or len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float().cpu().numpy()
        y_train = train_labels.float().cpu().numpy()
        X_test = test_features.float().cpu().numpy().reshape(1, -1)

        if task_type == "clf":
            y_int = (y_train > 0).astype(int)
            if len(np.unique(y_int)) < 2:
                return float(y_int[0])
            model = LGBMClassifier(**self.params)
            model.fit(X_train, y_int)
            return float(model.predict_proba(X_test)[0, 1])
        else:
            model = LGBMRegressor(**self.params)
            model.fit(X_train, y_train)
            return float(model.predict(X_test)[0])
