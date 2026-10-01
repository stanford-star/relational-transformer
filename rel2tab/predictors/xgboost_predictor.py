"""XGBoost predictor for the rel2tab pipeline.

Fits a fresh XGBoost model per prediction (per test row, per context size),
mirroring the Ridge / LGBM predictors.  The rel2tab in-context regime is
extreme low-data: at the swept context sizes the visible labeled set is only
~12-400 rows, while the precomputed feature sets range from 8 (SQL) to ~450
(RDBLearn on rel-event) columns.  Gradient-boosted trees with library defaults
(``max_depth=6``, ``n_estimators=100``, no regularization) overfit badly here,
so the configs are shallow and heavily regularized.

HP split (mirrors Ridge's alpha_clf / alpha_reg): the config carries SEPARATE
hyperparameter sets for classification and regression, since the best tree
shape differs by task type.  These two global HP sets are tuned on the
VALIDATION split only and shared across all tasks within each task type — no
per-task tuning, no test leakage.

``early_stopping_frac`` (per task type) optionally holds out a slice of the
in-context labels to early-stop on the number of boosting rounds; 0 disables it
(plain fixed-round fit).

Trees are scale-invariant, so (unlike Ridge) no feature standardization is
applied — matching the LGBM predictor.
"""

from dataclasses import dataclass

import numpy as np

from rel2tab.predictor import Predictor


@dataclass
class XGBoostHP:
    """One global hyperparameter set (used for either clf or reg).

    Args:
        n_estimators: max boosting rounds (upper bound when early stopping on).
        max_depth: tree depth.  2-4 is the sweet spot for <=400 rows.
        learning_rate: shrinkage.  Smaller is steadier; pairs with more rounds.
        min_child_weight: min sum of instance weight (hessian) per leaf —
            higher = stronger regularization (fewer, larger leaves).
        subsample: row subsampling per tree (stochastic regularization).
        colsample_bytree: column subsampling per tree.  Important for the
            high-dim RDBLearn rel-event features (~450 cols, few rows).
        reg_lambda: L2 penalty on leaf weights.
        reg_alpha: L1 penalty on leaf weights.
        early_stopping_frac: if > 0, hold out this fraction of the in-context
            labels (stratified for clf) to early-stop on n_estimators.
    """

    n_estimators: int
    max_depth: int
    learning_rate: float
    min_child_weight: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float
    early_stopping_frac: float

    def xgb_params(self, n_jobs):
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "reg_alpha": self.reg_alpha,
            "tree_method": "hist",
            "n_jobs": n_jobs,
            "verbosity": 0,
        }


@dataclass
class XGBoostPredictorConfig:
    """Config for XGBoostPredictor.

    Carries two global HP sets — ``clf`` and ``reg`` — analogous to the Ridge
    predictor's ``alpha_clf`` / ``alpha_reg`` split.

    Args:
        clf: HP set for classification tasks.
        reg: HP set for regression tasks.
        n_jobs: XGBoost threads per fit.  Kept to 1 because the outer per-item
            loop is what we parallelize across (via SLURM/DDP workers);
            per-fit threading on tiny data mostly adds overhead.
    """

    clf: XGBoostHP
    reg: XGBoostHP
    n_jobs: int

    def build(self):
        return XGBoostPredictor(clf=self.clf, reg=self.reg, n_jobs=self.n_jobs)


class XGBoostPredictor(Predictor):
    """Fit an XGBoost model per prediction (separate clf / reg HP sets)."""

    def __init__(self, clf, reg, n_jobs):
        self.clf_hp = clf
        self.reg_hp = reg
        self.n_jobs = n_jobs

    def predict(self, train_features, train_labels, test_features, task_type):
        from xgboost import XGBClassifier, XGBRegressor

        if train_features is None or len(train_labels) < 2:
            return 0.5 if task_type == "clf" else 0.0

        X_train = train_features.float().cpu().numpy()
        y_train = train_labels.float().cpu().numpy()
        X_test = test_features.float().cpu().numpy().reshape(1, -1)

        if task_type == "clf":
            hp = self.clf_hp
            y_int = (y_train > 0).astype(int)
            if len(np.unique(y_int)) < 2:
                return float(y_int[0])
            # scale_pos_weight = (#neg / #pos) to counter class imbalance.
            n_pos = int(y_int.sum())
            n_neg = int(len(y_int) - n_pos)
            spw = (n_neg / n_pos) if n_pos > 0 else 1.0
            model = XGBClassifier(
                **hp.xgb_params(self.n_jobs),
                objective="binary:logistic",
                eval_metric="logloss",
                scale_pos_weight=spw,
            )
            self._fit(model, X_train, y_int, hp.early_stopping_frac, stratify=True)
            return float(model.predict_proba(X_test)[0, 1])
        else:
            hp = self.reg_hp
            model = XGBRegressor(
                **hp.xgb_params(self.n_jobs),
                objective="reg:squarederror",
                eval_metric="mae",
            )
            self._fit(model, X_train, y_train, hp.early_stopping_frac, stratify=False)
            return float(model.predict(X_test)[0])

    def _fit(self, model, X, y, frac, stratify):
        """Fit with optional early stopping on a held-out slice of X.

        When ``frac == 0``, a plain fixed-round fit is used.
        """
        n = len(y)
        # Need enough rows on both sides for a meaningful holdout; also require
        # both classes present in train+val for clf.
        n_val = round(frac * n) if frac > 0 else 0
        if n_val < 1 or (n - n_val) < 2:
            model.set_params(early_stopping_rounds=None)
            model.fit(X, y)
            return

        rng = np.random.RandomState(0)
        if stratify and len(np.unique(y)) == 2:
            idx_val = []
            for cls in (0, 1):
                cls_idx = np.where(y == cls)[0]
                k = max(1, round(frac * len(cls_idx)))
                k = min(k, len(cls_idx) - 1) if len(cls_idx) > 1 else 0
                if k > 0:
                    idx_val.extend(rng.choice(cls_idx, size=k, replace=False).tolist())
            idx_val = np.array(sorted(idx_val), dtype=int)
        else:
            perm = rng.permutation(n)
            idx_val = perm[:n_val]

        if len(idx_val) < 1 or (n - len(idx_val)) < 2:
            model.set_params(early_stopping_rounds=None)
            model.fit(X, y)
            return

        mask = np.ones(n, dtype=bool)
        mask[idx_val] = False
        X_tr, y_tr = X[mask], y[mask]
        X_va, y_va = X[~mask], y[~mask]
        # Degenerate clf holdout (single class on either side): skip ES.
        if stratify and (len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2):
            model.set_params(early_stopping_rounds=None)
            model.fit(X, y)
            return

        model.set_params(early_stopping_rounds=20)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
