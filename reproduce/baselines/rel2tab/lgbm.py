import numpy as np


def _fit_predict(X_train, y_train, X_test, task_type, params):
    from lightgbm import LGBMClassifier, LGBMRegressor

    if task_type == "clf":
        y_int = (y_train > 0).astype(int)
        if len(np.unique(y_int)) < 2:
            return float(y_int[0])
        model = LGBMClassifier(**params)
        model.fit(X_train, y_int)
        return float(model.predict_proba(X_test)[0, 1])
    model = LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return float(model.predict(X_test)[0])


class LGBMPredictor:
    def __init__(self, n_jobs):
        self.n_jobs = n_jobs
        self.params = dict(n_jobs=1, verbose=-1)

    def _prep(self, train_features, train_labels, test_features, task_type):
        if len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0
        return (
            train_features.float().cpu().numpy(),
            np.nan_to_num(train_labels.float().cpu().numpy(), nan=0.0),
            test_features.float().cpu().numpy().reshape(1, -1),
            task_type,
        )

    def predict_batch(self, work_items):
        from joblib import Parallel, delayed

        results = [None] * len(work_items)
        jobs = []
        for i, (tf, tl, xf, tt) in enumerate(work_items):
            prepped = self._prep(tf, tl, xf, tt)
            if isinstance(prepped, tuple):
                jobs.append((i, prepped))
            else:
                results[i] = prepped
        fitted = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_predict)(*p, self.params) for _, p in jobs
        )
        for (i, _), v in zip(jobs, fitted):
            results[i] = v
        return results
