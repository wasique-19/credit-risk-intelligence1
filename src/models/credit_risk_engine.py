from src.models.decision_engine import make_credit_decision


def predict_credit_risk(model, X):
    """Generate default probability, risk grade, and business decision."""

    probabilities = model.predict_proba(X)[:, 1]

    results = []

    for probability in probabilities:
        decision = make_credit_decision(probability)
        results.append(decision)

    return results

def add_shap_explanation(
    shap_values,
    feature_names,
    row_index=0,
    top_n=5,
):
    """Add top positive and negative SHAP contributors."""

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
        "positive_contributors": positive,
        "negative_contributors": negative,
    }

def build_credit_risk_result(
    model,
    X,
    shap_values,
    feature_names,
    row_index=0,
    top_n=5,
):
    """Build a complete applicant-level credit risk result."""

    probability = float(model.predict_proba(X.iloc[[row_index]])[:, 1][0])

    result = make_credit_decision(probability)

    explanation = add_shap_explanation(
        shap_values,
        feature_names,
        row_index=row_index,
        top_n=top_n,
    )

    result.update(explanation)

    return result