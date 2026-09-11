# Portfolio Risk Analysis

## Purpose

The portfolio analysis layer converts model-predicted default probabilities
into risk grades and business decisions for a group of applicants.

This provides a portfolio-level view of credit risk instead of evaluating
only one applicant at a time.

## Validation Portfolio

The analysis was performed on the validation set containing **61,503
applicants**.

### Risk Grade Distribution

| Risk Grade | Applicants |
|---|---:|
| A | 29,033 |
| B | 16,796 |
| C | 10,858 |
| D | 3,159 |
| E | 1,657 |

### Business Decision Distribution

| Decision | Applicants |
|---|---:|
| APPROVE | 29,033 |
| MANUAL_REVIEW | 27,654 |
| REJECT | 4,816 |

## Portfolio Risk Metrics

- Average predicted default probability: **8.0413%**
- Median predicted default probability: **5.3339%**
- Highest predicted default probability: **74.8533%**

## Interpretation

The portfolio has an average predicted default probability of approximately
8.04%.

The median probability is lower at approximately 5.33%, indicating that a
smaller group of higher-risk applicants increases the portfolio average.

Approximately 45% of validation applicants fall into the manual-review
decision category under the current business rules.

## Important Note

The risk-grade and decision thresholds are initial portfolio/demo business
rules. They should not be treated as production credit-policy thresholds.

Production thresholds should consider expected loss, business costs,
approval targets, regulatory requirements, and operational review capacity.