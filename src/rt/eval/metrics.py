import sklearn.metrics as M


def metric_for(task_type: str, labels, preds) -> tuple[str, float]:
    if task_type == "reg":
        return "mae", float(M.mean_absolute_error(labels, preds))
    return "roc_auc", float(M.roc_auc_score((labels > 0).astype(int), preds))
