from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate


def cross_validate_model(
    model,
    X,
    y,
    n_splits=3,
    random_state=42,
):
    """Evaluate a classification model using stratified cross-validation."""

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    results = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
    )

    return {
        "roc_auc_mean": results["test_roc_auc"].mean(),
        "roc_auc_std": results["test_roc_auc"].std(),
        "pr_auc_mean": results["test_pr_auc"].mean(),
        "pr_auc_std": results["test_pr_auc"].std(),
    }