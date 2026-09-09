from sklearn.metrics import roc_auc_score, average_precision_score


def evaluate_model(model, X_valid, y_valid):
    """Evaluate a binary classification model."""

    probabilities = model.predict_proba(X_valid)[:, 1]

    roc_auc = roc_auc_score(y_valid, probabilities)
    pr_auc = average_precision_score(y_valid, probabilities)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }