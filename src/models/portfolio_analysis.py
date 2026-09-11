import pandas as pd

from src.models.decision_engine import assign_risk_grade, assign_decision


def segment_portfolio(probabilities):
    """Convert default probabilities into risk grades and decisions."""

    results = pd.DataFrame({
        "default_probability": probabilities,
    })

    results["risk_grade"] = results["default_probability"].apply(
        assign_risk_grade
    )

    results["decision"] = results["default_probability"].apply(
        assign_decision
    )

    return results

def summarize_portfolio(portfolio):
    """Summarize risk grades and business decisions."""

    total_applicants = len(portfolio)

    grade_counts = portfolio["risk_grade"].value_counts().sort_index()
    decision_counts = portfolio["decision"].value_counts()

    return {
        "total_applicants": total_applicants,
        "grade_counts": grade_counts.to_dict(),
        "decision_counts": decision_counts.to_dict(),
    }

def save_portfolio_analysis(portfolio, filename="portfolio_risk_analysis.csv"):
    """Save portfolio risk segmentation to the processed data directory."""

    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename
    portfolio.to_csv(output_path, index=False)

    return output_path