"""Identity predictor — returns the test feature directly as the prediction.

Designed for use with featurizers that produce a 1-d "prediction" feature
(e.g. SQLFeaturizer in ``mode="predictions"``).
"""

from dataclasses import dataclass

from rt.rel2tab.predictor import Predictor


@dataclass
class IdentityPredictorConfig:
    def build(self):
        return IdentityPredictor()


class IdentityPredictor(Predictor):
    """Return ``test_features[0]`` as the prediction.

    Falls back to 0.5 (clf) or 0.0 (reg) when test features are unavailable.
    """

    def predict(self, train_features, train_labels, test_features, task_type):
        if test_features is not None and len(test_features) > 0:
            return test_features[0].item()
        return 0.5 if task_type == "clf" else 0.0
