from dataclasses import dataclass

import numpy as np

from rt.rel2tab.predictor import Predictor


@dataclass
class LinearPredictorConfig:
    """Config for LinearPredictor (no fields needed)."""

    def build(self):
        return LinearPredictor()


class LinearPredictor(Predictor):
    """Fit an sklearn linear (regression) or logistic (classification) model."""

    def predict(self, train_features, train_labels, test_features, task_type):
        from sklearn.linear_model import LinearRegression, LogisticRegression

        if train_features is None or len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float().cpu().numpy()
        y_train = train_labels.float().cpu().numpy()
        X_test = test_features.float().cpu().numpy().reshape(1, -1)

        if task_type == "clf":
            y_int = (y_train > 0).astype(int)
            if len(np.unique(y_int)) < 2:
                return float(y_int[0])
            model = LogisticRegression(max_iter=200, solver="lbfgs")
            model.fit(X_train, y_int)
            return float(model.predict_proba(X_test)[0, 1])
        else:
            model = LinearRegression()
            model.fit(X_train, y_train)
            return float(model.predict(X_test)[0])
