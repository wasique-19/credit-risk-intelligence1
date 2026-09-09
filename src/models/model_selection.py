def select_champion_model(results):
    """Select the model with the highest PR-AUC."""

    if not results:
        raise ValueError("No model results provided")

    return max(results, key=lambda x: x["pr_auc"])