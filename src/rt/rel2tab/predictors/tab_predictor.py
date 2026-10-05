import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

import numpy as np

from rt.rel2tab.predictor import Predictor


@dataclass
class TabPredictorConfig:
    """Config for TabPredictor (unified TabICL/TabPFN)."""

    model: Literal["tabicl", "tabpfn"]
    num_workers: int

    def build(self):
        return TabPredictor(model=self.model, num_workers=self.num_workers)


class TabPredictor(Predictor):
    """Fit a tabular foundation model on train features and predict for test row.

    Supports TabICL and TabPFN via the ``model`` argument. Uses thread-local
    model instances for concurrent prediction via ``predict_batch``.
    """

    def __init__(self, model, num_workers):
        import torch

        self.model = model
        self.num_workers = num_workers
        self._device = f"cuda:{torch.cuda.current_device()}"
        self._local = threading.local()

    def _get_thread_models(self):
        """Return thread-local (clf, reg) pair, creating them if needed."""
        if not hasattr(self._local, "clf"):
            if self.model == "tabicl":
                from tabicl import TabICLClassifier, TabICLRegressor

                self._local.clf = TabICLClassifier(device=self._device, use_amp=False)
                self._local.reg = TabICLRegressor(device=self._device, use_amp=False)
            elif self.model == "tabpfn":
                from tabpfn import TabPFNClassifier, TabPFNRegressor

                self._local.clf = TabPFNClassifier(device=self._device)
                self._local.reg = TabPFNRegressor(device=self._device)
        return self._local.clf, self._local.reg

    def _predict_one(self, train_features, train_labels, test_features, task_type):
        """Single prediction using thread-local models."""
        if train_features is None or len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float().cpu().numpy()
        y_train = train_labels.float().cpu().numpy()
        X_test = test_features.float().cpu().numpy().reshape(1, -1)

        if np.all(X_train == X_train[0]):
            return float(y_train.mean())

        clf, reg = self._get_thread_models()

        if task_type == "clf":
            y_int = (y_train > 0).astype(int)
            if len(np.unique(y_int)) < 2:
                return float(y_int[0])
            clf.fit(X_train, y_int)
            return float(clf.predict_proba(X_test)[0, 1])
        else:
            reg.fit(X_train, y_train)
            return float(reg.predict(X_test)[0])

    def predict(self, train_features, train_labels, test_features, task_type):
        return self._predict_one(train_features, train_labels, test_features, task_type)

    def predict_batch(self, work_items):
        """Predict many items concurrently using thread-local model copies.

        Args:
            work_items: list of (train_features, train_labels, test_features,
                task_type) tuples.

        Returns:
            list of scalar float predictions, one per work item.
        """
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [
                pool.submit(self._predict_one, tf, tl, xf, tt)
                for tf, tl, xf, tt in work_items
            ]
            return [f.result() for f in futures]
