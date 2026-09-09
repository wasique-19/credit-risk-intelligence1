from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_model(model, X_valid, y_valid, threshold=0.5):
    """Evaluate a binary classification model."""

    probabilities = model.predict_proba(X_valid)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_valid,
        predictions,
    ).ravel()

    return {
        "roc_auc": roc_auc_score(y_valid, probabilities),
        "pr_auc": average_precision_score(y_valid, probabilities),
        "precision": precision_score(
            y_valid,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_valid,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_valid,
            predictions,
            zero_division=0,
        ),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }