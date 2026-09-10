import numpy as np
import shap


def create_tree_explainer(model):
    """Create a SHAP TreeExplainer for a tree-based classifier."""
    classifier = model.named_steps["classifier"]
    return shap.TreeExplainer(classifier)


def calculate_shap_values(model, X_transformed):
    """Calculate SHAP values for transformed input data."""
    explainer = create_tree_explainer(model)
    return explainer.shap_values(X_transformed)


def calculate_feature_importance(shap_values, feature_names):
    """Calculate mean absolute SHAP importance for each feature."""
    importance = np.abs(shap_values).mean(axis=0)

    result = sorted(
        zip(feature_names, importance),
        key=lambda x: x[1],
        reverse=True,
    )

    return result