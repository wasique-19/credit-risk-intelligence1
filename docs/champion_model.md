\# Champion Model — Calibrated LightGBM



\## Selected Model



\*\*Champion:\*\* Calibrated LightGBM



The final champion model is a LightGBM classifier wrapped with probability

calibration using `CalibratedClassifierCV`.



\## Model Selection



Models were compared using PR-AUC as the primary metric because the Home Credit

dataset has a highly imbalanced target.



| Model | ROC-AUC | PR-AUC |

|---|---:|---:|

| Logistic Regression | 0.7493 | 0.2280 |

| Random Forest | 0.7368 | 0.2175 |

| XGBoost | 0.7622 | 0.2532 |

| LightGBM | 0.7614 | 0.2550 |

| Calibrated LightGBM | 0.7627 | 0.2552 |



\## Stability Validation



The underlying LightGBM model was evaluated using 3-fold Stratified

Cross-Validation.



\- Mean ROC-AUC: \*\*0.7584\*\*

\- ROC-AUC standard deviation: \*\*0.0018\*\*

\- Mean PR-AUC: \*\*0.2418\*\*

\- PR-AUC standard deviation: \*\*0.0032\*\*



The low standard deviation indicates stable performance across folds.



\## Probability Calibration



The LightGBM model was calibrated using `CalibratedClassifierCV` with

3-fold cross-validation and sigmoid calibration.



Calibration results showed that predicted probabilities were reasonably aligned

with observed default rates, particularly in the higher-risk probability bins.



\## Decision Threshold



A threshold of \*\*0.15\*\* produced approximately:



\- Precision: \*\*24.81%\*\*

\- Recall: \*\*42.09%\*\*

\- F1: \*\*31.22%\*\*



This threshold is \*\*not locked as the final production threshold\*\*.



The production decision threshold will be selected later using the business

decision engine and cost-sensitive analysis.



\## Conclusion



Calibrated LightGBM is selected as the champion model because it provides the

best observed PR-AUC while maintaining strong ROC-AUC performance and

well-calibrated probability estimates.



The model is ready to move into the explainability and decision-engine stages.

