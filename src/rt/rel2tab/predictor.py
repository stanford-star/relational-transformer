from abc import ABC, abstractmethod


class Predictor(ABC):
    """Maps train rows and a test-row feature into a single scalar prediction.

    Called once per (batch-item, context-size) after the featurizer has
    selected/transformed the visible rows.
    """

    @abstractmethod
    def predict(self, train_features, train_labels, test_features, task_type):
        """Produce a prediction for one target row.

        Args:
            train_features: (num_train, d_feat) Tensor, or None if the
                featurizer does not produce features.
            train_labels: 1-D float Tensor of train labels (may be empty).
            test_features: (d_feat,) Tensor, or None.
            task_type: "clf" for binary classification, "reg" for regression.

        Returns:
            Scalar float: probability in [0, 1] for clf, real value for reg.
            Convention when no train data is available: 0.5 for clf, 0.0 for reg.
        """
