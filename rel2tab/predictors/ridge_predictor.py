"""Ridge predictor — regularized linear model for few-shot tabular prediction.

Uses Ridge regression (L2-regularized) for regression and LogisticRegression
with L2 penalty for classification.  Designed for the rel2tab pipeline where
training sets per prediction are very small (2-50 rows).
"""

from dataclasses import dataclass

import numpy as np

from rel2tab.predictor import Predictor


@dataclass
class RidgePredictorConfig:
    alpha_clf: float
    alpha_reg: float

    def build(self):
        return RidgePredictor(alpha_clf=self.alpha_clf, alpha_reg=self.alpha_reg)


class RidgePredictor(Predictor):
    """Fit a regularized linear model per prediction."""

    def __init__(self, alpha_clf, alpha_reg):
        self.alpha_clf = alpha_clf
        self.alpha_reg = alpha_reg

    def predict(self, train_features, train_labels, test_features, task_type):
        from sklearn.linear_model import LogisticRegression, Ridge

        if train_features is None or len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float().cpu().numpy()
        y_train = train_labels.float().cpu().numpy()
        X_test = test_features.float().cpu().numpy().reshape(1, -1)

        # Standardize features to prevent numerical blowup
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        if task_type == "clf":
            y_int = (y_train > 0).astype(int)
            if len(np.unique(y_int)) < 2:
                return float(y_int[0])
            model = LogisticRegression(
                C=1.0 / max(self.alpha_clf, 1e-8),
                max_iter=500,
                solver="lbfgs",
            )
            model.fit(X_train, y_int)
            return float(model.predict_proba(X_test)[0, 1])
        else:
            model = Ridge(alpha=self.alpha_reg)
            model.fit(X_train, y_train)
            return float(model.predict(X_test)[0])
