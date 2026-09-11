def assign_risk_grade(default_probability):
    """Assign a risk grade based on predicted default probability."""

    if not 0 <= default_probability <= 1:
        raise ValueError("default_probability must be between 0 and 1")

    if default_probability < 0.05:
        return "A"
    elif default_probability < 0.10:
        return "B"
    elif default_probability < 0.20:
        return "C"
    elif default_probability < 0.30:
        return "D"
    else:
        return "E"


def assign_decision(default_probability):
    """Convert default probability into a business decision."""

    if not 0 <= default_probability <= 1:
        raise ValueError("default_probability must be between 0 and 1")

    if default_probability < 0.05:
        return "APPROVE"
    elif default_probability < 0.20:
        return "MANUAL_REVIEW"
    else:
        return "REJECT"

def make_credit_decision(default_probability):
    """Return probability, risk grade, and business decision."""

    return {
        "default_probability": default_probability,
        "risk_grade": assign_risk_grade(default_probability),
        "decision": assign_decision(default_probability),
    }