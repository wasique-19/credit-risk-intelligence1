\# Credit Risk Intelligence



An end-to-end machine learning system for predicting loan default risk, assigning risk grades and business decisions, and explaining individual predictions using SHAP.



\## Project Overview



This project uses the \*\*Home Credit Default Risk\*\* dataset to build a practical credit-risk prediction pipeline.



The system answers three business questions:



1\. What is the applicant's probability of default?

2\. What risk grade should the applicant receive?

3\. Should the application be approved, manually reviewed, or rejected?



The final system includes:



\* Feature engineering

\* Leakage-safe preprocessing

\* Multiple ML model benchmarks

\* Probability calibration

\* Champion model selection

\* SHAP explainability

\* Business decision engine

\* Portfolio risk segmentation

\* FastAPI prediction API

\* Streamlit dashboard

\* Docker deployment

\* Automated validation tests



\## Architecture



```text

Home Credit Dataset

&#x20;       │

&#x20;       ▼

Data Loading \& Validation

&#x20;       │

&#x20;       ▼

Feature Engineering

&#x20;       │

&#x20;       ▼

Leakage-Safe Preprocessing

&#x20;       │

&#x20;       ▼

Model Benchmarking

&#x20;       │

&#x20;       ├── Logistic Regression

&#x20;       ├── Random Forest

&#x20;       ├── XGBoost

&#x20;       └── LightGBM

&#x20;       │

&#x20;       ▼

Calibrated LightGBM

&#x20;       │

&#x20;       ├── Default Probability

&#x20;       ├── Risk Grade

&#x20;       ├── Business Decision

&#x20;       └── SHAP Explanation

&#x20;               │

&#x20;       ┌───────┴────────┐

&#x20;       ▼                ▼

&#x20;   FastAPI          Streamlit

&#x20;       │                │

&#x20;       └───────┬────────┘

&#x20;               ▼

&#x20;         Credit Risk Result

```



\## Dataset



The project uses the \*\*Home Credit Default Risk\*\* dataset.



Primary files:



\* `application\_train.csv`

\* `application\_test.csv`



The training dataset contains \*\*307,511 applicants and 122 columns\*\*, including the `TARGET` default indicator.



The default rate is approximately \*\*8.07%\*\*, making this an imbalanced classification problem.



Raw and processed datasets are intentionally excluded from Git because of their size.



\## Feature Engineering



The project creates additional risk-related features including:



\* Loan-to-income ratio

\* Annuity-to-income ratio

\* Credit-to-goods ratio

\* Applicant age

\* Clean employment duration

\* Employment-to-age ratio

\* Income per child

\* Income per family member

\* Credit minus income

\* Annuity minus income



Feature engineering is implemented in:



```text

src/features/feature\_engineering.py

```



\## Model Benchmark



| Model                   |    ROC-AUC |     PR-AUC |

| ----------------------- | ---------: | ---------: |

| Logistic Regression     |     0.7493 |     0.2280 |

| Random Forest           |     0.7368 |     0.2175 |

| XGBoost                 |     0.7622 |     0.2532 |

| LightGBM                |     0.7614 |     0.2550 |

| \*\*Calibrated LightGBM\*\* | \*\*0.7627\*\* | \*\*0.2552\*\* |



Because the dataset is highly imbalanced, \*\*PR-AUC is considered alongside ROC-AUC\*\* rather than relying on accuracy alone.



\## Model Validation



A stratified 3-fold cross-validation analysis was performed on the LightGBM baseline.



Results:



\* Mean ROC-AUC: \*\*0.7584\*\*

\* ROC-AUC standard deviation: \*\*0.0018\*\*

\* Mean PR-AUC: \*\*0.2418\*\*

\* PR-AUC standard deviation: \*\*0.0032\*\*



The low standard deviations indicate relatively stable model performance across folds.



\## Probability Calibration



The selected LightGBM model was calibrated using `CalibratedClassifierCV` with sigmoid calibration.



The calibrated model achieved:



\* ROC-AUC: \*\*0.7627\*\*

\* PR-AUC: \*\*0.2552\*\*



Calibration was evaluated using predicted-risk versus observed-default bins.



\## Risk Decision Engine



The model probability is converted into a business-friendly risk grade:



| Default Probability | Risk Grade |

| ------------------: | :--------: |

|                < 5% |      A     |

|             5%–<10% |      B     |

|            10%–<20% |      C     |

|            20%–<30% |      D     |

|               ≥ 30% |      E     |



Business decisions:



| Default Probability | Decision      |

| ------------------: | ------------- |

|                < 5% | APPROVE       |

|             5%–<20% | MANUAL\_REVIEW |

|               ≥ 20% | REJECT        |



These thresholds are intentionally implemented as a simple business rule and should be recalibrated using real business costs, approval constraints, and regulatory requirements in a production environment.



\## Explainability with SHAP



The system uses \*\*SHAP TreeExplainer\*\* to explain individual LightGBM predictions.



For each applicant, the API returns:



\* Top positive risk contributors

\* Top negative risk contributors

\* SHAP values for each contributor



Example applicant:



```text

Applicant ID: 396899

Default Probability: 6.31%

Risk Grade: B

Decision: MANUAL\_REVIEW

```



Example positive contributors:



```text

EXT\_SOURCE\_3

EMPLOYMENT\_AGE\_RATIO

EXT\_SOURCE\_1

DAYS\_EMPLOYED\_CLEAN

CODE\_GENDER\_F

```



Example negative contributors:



```text

AMT\_GOODS\_PRICE

NAME\_EDUCATION\_TYPE\_Higher education

AMT\_REQ\_CREDIT\_BUREAU\_QRT

EXT\_SOURCE\_2

NAME\_EDUCATION\_TYPE\_Secondary / secondary special

```



SHAP values explain model behavior and should \*\*not be interpreted as causal effects\*\*.



\## Portfolio Risk Segmentation



On the validation portfolio of \*\*61,503 applicants\*\*:



| Risk Grade | Applicants |

| ---------- | ---------: |

| A          |     29,033 |

| B          |     16,796 |

| C          |     10,858 |

| D          |      3,159 |

| E          |      1,657 |



Business decisions:



| Decision      | Applicants | Share |

| ------------- | ---------: | ----: |

| APPROVE       |     29,033 | 47.2% |

| MANUAL\_REVIEW |     27,654 | 45.0% |

| REJECT        |      4,816 |  7.8% |



Average predicted default probability was approximately \*\*8.04%\*\*.



\## FastAPI



The project provides a REST API for applicant-level prediction.



Start the API:



```powershell

uvicorn src.api.main:app --reload

```



API:



```text

http://127.0.0.1:8000

```



Swagger documentation:



```text

http://127.0.0.1:8000/docs

```



Health check:



```text

GET /health

```



Prediction:



```text

POST /predict

```



Example request:



```json

{

&#x20; "applicant\_id": 396899

}

```



Example response:



```json

{

&#x20; "applicant\_id": 396899,

&#x20; "default\_probability": 0.06305131775793747,

&#x20; "risk\_grade": "B",

&#x20; "decision": "MANUAL\_REVIEW"

}

```



The prediction endpoint also returns SHAP-based positive and negative contributors.



\## Streamlit Dashboard



The project includes an interactive Streamlit dashboard.



Start it with:



```powershell

streamlit run src/dashboard/app.py

```



The dashboard displays:



\* Default probability

\* Risk grade

\* Business decision

\* Risk interpretation

\* Positive SHAP contributors

\* Negative SHAP contributors



The dashboard communicates with the FastAPI backend.



\## Docker



The FastAPI service can be packaged as a Docker image.



Build:



```powershell

docker build -t credit-risk-api .

```



Run:



```powershell

docker run -d --name credit-risk-api-container -p 8000:8000 credit-risk-api

```



Health check:



```powershell

curl http://localhost:8000/health

```



The Docker image contains the application code, calibrated model, raw application data required by the current API implementation, and Python dependencies.



\## Testing



Automated tests are included for:



\* Decision-engine boundaries

\* Data validation

\* Feature engineering



Run:



```powershell

pytest -q

```



Current result:



```text

9 passed

```



Python syntax can also be checked with:



```powershell

python -m compileall src tests

```



\## Project Structure



```text

credit-risk-intelligence1/

│

├── data/

│   ├── raw/

│   └── processed/

│

├── notebooks/

│   ├── 01\_initial\_eda.ipynb

│   ├── 02\_feature\_engineering.ipynb

│   └── 03\_shap\_visualization.ipynb

│

├── src/

│   ├── api/

│   │   └── main.py

│   │

│   ├── dashboard/

│   │   └── app.py

│   │

│   ├── data/

│   │   ├── data\_loader.py

│   │   └── data\_validation.py

│   │

│   ├── features/

│   │   ├── data\_preparation.py

│   │   ├── feature\_engineering.py

│   │   └── preprocessing.py

│   │

│   └── models/

│       ├── baseline\_model.py

│       ├── random\_forest\_model.py

│       ├── xgboost\_model.py

│       ├── lightgbm\_model.py

│       ├── model\_evaluation.py

│       ├── threshold\_analysis.py

│       ├── calibration.py

│       ├── model\_selection.py

│       ├── cross\_validation.py

│       ├── shap\_explainability.py

│       ├── decision\_engine.py

│       ├── portfolio\_analysis.py

│       └── credit\_risk\_engine.py

│

├── tests/

├── models/

├── docs/

├── Dockerfile

├── pytest.ini

├── requirements.txt

└── README.md

```



\## Key Engineering Practices



This project demonstrates:



\* Stratified train/validation splitting

\* Leakage-safe preprocessing

\* Reusable sklearn pipelines

\* Class-imbalance awareness

\* Multiple model benchmarking

\* Cross-validation

\* Probability calibration

\* Threshold analysis

\* SHAP explainability

\* Modular Python architecture

\* REST API development

\* Interactive dashboard development

\* Automated testing

\* Docker containerization

\* Git/GitHub version control



\## Limitations



This is a portfolio project and should not be treated as a production lending system.



Important limitations include:



\* The model is trained primarily on the Home Credit application table.

\* Decision thresholds are illustrative business rules.

\* Model performance can change on new populations.

\* SHAP explanations describe model behavior, not causal relationships.

\* Production credit decisions require fairness, compliance, monitoring, governance, and human oversight.

\* Additional Home Credit relational tables could be incorporated for richer feature engineering.



\## Future Improvements



Potential next steps:



\* Integrate additional Home Credit tables

\* Hyperparameter optimization

\* More rigorous probability calibration

\* Cost-sensitive threshold optimization

\* Fairness and bias analysis

\* Model monitoring

\* Data drift detection

\* PostgreSQL integration

\* Authentication and API security

\* CI/CD pipeline

\* Cloud deployment



\## Author



\*\*Wasique\*\*



Machine Learning / Data Science Portfolio Project



GitHub: `wasique-19/credit-risk-intelligence1`
