from src.models.decision_engine import (
    assign_decision,
    assign_risk_grade,
    make_credit_decision,
)


def test_risk_grade_boundaries():
    assert assign_risk_grade(0.0499) == "A"
    assert assign_risk_grade(0.05) == "B"
    assert assign_risk_grade(0.10) == "C"
    assert assign_risk_grade(0.20) == "D"
    assert assign_risk_grade(0.30) == "E"


def test_decision_boundaries():
    assert assign_decision(0.0499) == "APPROVE"
    assert assign_decision(0.05) == "MANUAL_REVIEW"
    assert assign_decision(0.1999) == "MANUAL_REVIEW"
    assert assign_decision(0.20) == "REJECT"


def test_complete_credit_decision():
    result = make_credit_decision(0.063051)

    assert result["default_probability"] == 0.063051
    assert result["risk_grade"] == "B"
    assert result["decision"] == "MANUAL_REVIEW"