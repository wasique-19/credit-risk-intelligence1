# Unified Credit Risk Engine

## Purpose

The unified credit risk engine combines machine-learning predictions,
business decision rules, and SHAP explanations into a single applicant-level
risk result.

## Processing Flow

```text
Applicant Data
      ↓
Calibrated LightGBM
      ↓
Default Probability
      ↓
Risk Grade
      ↓
Business Decision
      ↓
SHAP Explanation