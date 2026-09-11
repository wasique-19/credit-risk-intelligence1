# Credit Risk Decision Engine

## Purpose

The decision engine converts a model's predicted default probability into:

1. Risk Grade
2. Business Decision

This creates a simple business layer on top of the machine-learning model.

## Risk Grade Rules

| Default Probability | Risk Grade |
|---|---|
| < 5% | A |
| 5% – <10% | B |
| 10% – <20% | C |
| 20% – <30% | D |
| ≥30% | E |

## Business Decision Rules

| Default Probability | Decision |
|---|---|
| < 5% | APPROVE |
| 5% – <20% | MANUAL_REVIEW |
| ≥20% | REJECT |

## Example

A calibrated model prediction of **6.3051%** produces:

- Risk Grade: **B**
- Decision: **MANUAL_REVIEW**

## Important Note

These thresholds are initial portfolio/demo business rules. They are not final production credit-policy thresholds.

A production system should determine thresholds using business costs, approval targets, expected losses, regulatory requirements, and calibration analysis.