"""Tuned global XGBoost hyperparameter sets for the precomputed-feature
in-context baselines (precomputed_{sql,rdblearn}_xgboost).

For each feature set (SQL / RDBLearn), two global HP sets — one for
classification, one for regression — shared across all RelBench tasks within
each task type (no per-task tuning). Tuned on the VALIDATION split only
(few-shot fit on in-context labels, scored on val targets; the test split is
never touched during tuning).

The SQL and RDBLearn feature sets are tuned separately because their feature
dimensionality differs by ~30x (SQL 8-15 vs RDBLearn 9-452), so the best tree
shape differs. Update by re-running the tuner per feature set and pasting the
winning configs.
"""

import json
import os

from rt.rel2tab.predictors.xgboost_predictor import XGBoostHP, XGBoostPredictorConfig

# Optional runtime override: if env var XGB_TUNED_JSON points to a JSON file,
# read tuned clf/reg HP sets from it. Lets a single job tune then eval with the
# just-found winners without a source edit/commit. JSON schema:
#   {"sql_features": {"clf": {<XGBoostHP fields>}, "reg": {...}},
#    "rdblearn_features": {"clf": {...}, "reg": {...}}}
_OVERRIDE_ENV = "XGB_TUNED_JSON"

# ===================== SQL features (val-tuned) =====================
SQL_TUNED_CLF = XGBoostHP(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.0,
    early_stopping_frac=0.0,
)
SQL_TUNED_REG = XGBoostHP(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.0,
    early_stopping_frac=0.0,
)

# =================== RDBLearn features (val-tuned) ===================
RDBLEARN_TUNED_CLF = XGBoostHP(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.0,
    early_stopping_frac=0.0,
)
RDBLEARN_TUNED_REG = XGBoostHP(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=5.0,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.0,
    early_stopping_frac=0.0,
)


def tuned_xgboost_config(features_subdir):
    """Return the val-tuned XGBoostPredictorConfig for a feature set.

    ``features_subdir`` is "sql_features" or "rdblearn_features".
    """
    override_path = os.environ.get(_OVERRIDE_ENV)
    if override_path and os.path.exists(override_path):
        with open(override_path) as f:
            blob = json.load(f)
        if features_subdir in blob:
            entry = blob[features_subdir]
            clf = XGBoostHP(**entry["clf"])
            reg = XGBoostHP(**entry["reg"])
            return XGBoostPredictorConfig(clf=clf, reg=reg, n_jobs=1)

    if features_subdir == "sql_features":
        clf, reg = SQL_TUNED_CLF, SQL_TUNED_REG
    elif features_subdir == "rdblearn_features":
        clf, reg = RDBLEARN_TUNED_CLF, RDBLEARN_TUNED_REG
    else:
        raise ValueError(
            f"No tuned XGBoost config for features_subdir={features_subdir!r}"
        )
    return XGBoostPredictorConfig(clf=clf, reg=reg, n_jobs=1)
