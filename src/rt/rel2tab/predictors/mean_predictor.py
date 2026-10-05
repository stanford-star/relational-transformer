from dataclasses import dataclass

from rt.rel2tab.predictor import Predictor


@dataclass
class MeanPredictorConfig:
    """Config for MeanPredictor (no fields needed)."""

    def build(self):
        return MeanPredictor()


class MeanPredictor(Predictor):
    """Predict the mean of training labels (ignores features)."""

    def predict(self, train_features, train_labels, test_features, task_type):
        if len(train_labels) == 0:
            return 0.5 if task_type == "clf" else 0.0
        return train_labels.mean().item()
