import numpy as np
import shap


def create_tree_explainer(model):
    """Create a SHAP TreeExplainer for a fitted tree-based pipeline."""
    classifier = model.named_steps["classifier"]
    return shap.TreeExplainer(classifier)


def calculate_shap_values(model, X_transformed):
    """Calculate SHAP values for transformed input data."""
    explainer = create_tree_explainer(model)
    shap_values = explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    return shap_values


def calculate_feature_importance(shap_values, feature_names):
    """Calculate mean absolute SHAP importance for each feature."""
    importance = np.abs(shap_values).mean(axis=0)

    result = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True,
    )

    return result


def explain_prediction(
    shap_values,
    feature_names,
    row_index=0,
    top_n=10,
):
    """Return top positive and negative SHAP contributors for one prediction."""
    row_values = shap_values[row_index]

    contributions = list(zip(feature_names, row_values))

    positive = sorted(
        [item for item in contributions if item[1] > 0],
        key=lambda x: x[1],
        reverse=True,
    )[:top_n]

    negative = sorted(
        [item for item in contributions if item[1] < 0],
        key=lambda x: x[1],
    )[:top_n]

    return {
        "positive": positive,
        "negative": negative,
    }