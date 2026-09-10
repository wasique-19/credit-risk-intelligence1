\# Day 18 — SHAP Explainability Analysis



\## Objective



SHAP (SHapley Additive exPlanations) was introduced to make the champion

LightGBM model more interpretable.



The analysis identifies which features have the largest overall influence on

the model's predictions.



\## Method



\- Model: Calibrated LightGBM

\- SHAP explainer: `shap.TreeExplainer`

\- Sample size: 1,000 validation observations

\- Transformed feature count: 256

\- Importance metric: mean absolute SHAP value



\## Top 10 Global Features



| Rank | Feature | Mean Absolute SHAP |

|---:|---|---:|

| 1 | EXT\_SOURCE\_3 | 0.359748 |

| 2 | EXT\_SOURCE\_2 | 0.315545 |

| 3 | EXT\_SOURCE\_1 | 0.154056 |

| 4 | CREDIT\_GOODS\_RATIO | 0.117354 |

| 5 | AMT\_GOODS\_PRICE | 0.091535 |

| 6 | CODE\_GENDER\_F | 0.074211 |

| 7 | AMT\_ANNUITY | 0.071688 |

| 8 | NAME\_EDUCATION\_TYPE\_Higher education | 0.071320 |

| 9 | DAYS\_ID\_PUBLISH | 0.052927 |

| 10 | NAME\_FAMILY\_STATUS\_Married | 0.045106 |



\## Interpretation



The three `EXT\_SOURCE` features are the strongest global contributors in the

sample. Several loan-related features, including `CREDIT\_GOODS\_RATIO` and

`AMT\_GOODS\_PRICE`, also have substantial influence.



Demographic and application-history features contribute as well, but with

lower average absolute SHAP impact than the leading external-source and

loan-related features.



\## Important Note



Mean absolute SHAP values measure \*\*feature importance\*\*, not the direction of

the effect.



A high SHAP importance does not by itself mean that a feature increases

default risk. Feature direction will be examined using individual SHAP values

and applicant-level explanations.



\## Next Step



The SHAP framework will later be extended to generate individual applicant

explanations for the decision engine and dashboard.

