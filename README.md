# Credit Risk Intelligence

**End-to-end machine learning system for loan default prediction, risk grading, business decisions, and SHAP-based explainability.**

---

## 🚀 Project Overview

This project builds an end-to-end **credit risk prediction system** using the **Home Credit Default Risk** dataset.

The system answers three key business questions:

1. **What is the applicant's probability of default?**
2. **What risk grade should the applicant receive?**
3. **Should the application be approved, manually reviewed, or rejected?**

### What this project includes

* **Feature engineering**
* **Leakage-safe preprocessing**
* **Multiple ML model benchmarking**
* **Probability calibration**
* **Champion model selection**
* **SHAP explainability**
* **Business decision engine**
* **Portfolio risk segmentation**
* **FastAPI prediction API**
* **Streamlit dashboard**
* **Docker deployment**
* **Automated testing**

---

## 🏗️ System Architecture

```text
Home Credit Dataset
        │
        ▼
Data Loading & Validation
        │
        ▼
Feature Engineering
        │
        ▼
Leakage-Safe Preprocessing
        │
        ▼
Model Benchmarking
        │
        ├── Logistic Regression
        ├── Random Forest
        ├── XGBoost
        └── LightGBM
        │
        ▼
Calibrated LightGBM
        │
        ├── Default Probability
        ├── Risk Grade
        ├── Business Decision
        └── SHAP Explanation
                │
        ┌───────┴────────┐
        ▼                ▼
     FastAPI         Streamlit
        │                │
        └───────┬────────┘
                ▼
        Credit Risk Result
```

---

## 📊 Dataset

The project uses the **Home Credit Default Risk** dataset.

Primary files:

* `application_train.csv`
* `application_test.csv`

Training dataset:

* **307,511 applicants**
* **122 columns**
* Default rate: **8.07%**

Because the dataset is highly imbalanced, **accuracy is not used as the primary evaluation metric**.

Raw datasets are excluded from Git because of their size.

---

## 🧠 Feature Engineering

The project creates domain-relevant credit-risk features including:

* **Loan-to-income ratio**
* **Annuity-to-income ratio**
* **Credit-to-goods ratio**
* **Applicant age**
* **Clean employment duration**
* **Employment-to-age ratio**
* **Income per child**
* **Income per family member**
* **Credit minus income**
* **Annuity minus income**

Implementation:

```text
src/features/feature_engineering.py
```

---

## 🏆 Model Benchmark

| Model                   |    ROC-AUC |     PR-AUC |
| ----------------------- | ---------: | ---------: |
| Logistic Regression     |     0.7493 |     0.2280 |
| Random Forest           |     0.7368 |     0.2175 |
| XGBoost                 |     0.7622 |     0.2532 |
| LightGBM                |     0.7614 |     0.2550 |
| **Calibrated LightGBM** | **0.7627** | **0.2552** |

### Champion Model

**Calibrated LightGBM**

The model was selected using **PR-AUC**, which is especially useful for this imbalanced classification problem.

---

## 🔬 Model Validation

A stratified **3-fold cross-validation** analysis was performed.

| Metric  |       Mean |    Std |
| ------- | ---------: | -----: |
| ROC-AUC | **0.7584** | 0.0018 |
| PR-AUC  | **0.2418** | 0.0032 |

The relatively low standard deviation indicates **stable model performance across folds**.

---

## 🎯 Probability Calibration

The LightGBM model was calibrated using:

```python
CalibratedClassifierCV
```

with **sigmoid calibration**.

Final calibrated performance:

* **ROC-AUC: 0.7627**
* **PR-AUC: 0.2552**

Calibration helps make predicted probabilities more useful for downstream **risk-based decision making**.

---

## 💳 Risk Decision Engine

Predicted default probability is converted into a business-friendly risk grade.

### Risk Grades

| Default Probability | Risk Grade |
| ------------------: | :--------: |
|              `< 5%` |    **A**   |
|         `5% – <10%` |    **B**   |
|        `10% – <20%` |    **C**   |
|        `20% – <30%` |    **D**   |
|             `≥ 30%` |    **E**   |

### Business Decisions

| Default Probability | Decision          |
| ------------------: | ----------------- |
|              `< 5%` | **APPROVE**       |
|         `5% – <20%` | **MANUAL_REVIEW** |
|             `≥ 20%` | **REJECT**        |

> These thresholds are illustrative business rules and should be optimized using real business costs, approval constraints, regulatory requirements, and historical outcomes before production use.

---

## 🔎 SHAP Explainability

The system uses **SHAP TreeExplainer** to explain individual LightGBM predictions.

For each applicant, the system returns:

* **Top positive risk contributors**
* **Top negative risk contributors**
* **SHAP values**

### Example Applicant

**Applicant ID:** `396899`

**Default Probability:** `6.31%`

**Risk Grade:** `B`

**Decision:** `MANUAL_REVIEW`

### Positive Risk Contributors

1. `EXT_SOURCE_3`
2. `EMPLOYMENT_AGE_RATIO`
3. `EXT_SOURCE_1`
4. `DAYS_EMPLOYED_CLEAN`
5. `CODE_GENDER_F`

### Negative Risk Contributors

1. `AMT_GOODS_PRICE`
2. `NAME_EDUCATION_TYPE_Higher education`
3. `AMT_REQ_CREDIT_BUREAU_QRT`
4. `EXT_SOURCE_2`
5. `NAME_EDUCATION_TYPE_Secondary / secondary special`

> **Important:** SHAP values explain model behavior and should not be interpreted as causal effects.

---

## 📈 Portfolio Risk Segmentation

Validation portfolio size:

**61,503 applicants**

### Risk Grade Distribution

| Risk Grade | Applicants |
| ---------- | ---------: |
| **A**      |     29,033 |
| **B**      |     16,796 |
| **C**      |     10,858 |
| **D**      |      3,159 |
| **E**      |      1,657 |

### Business Decision Distribution

| Decision          | Applicants | Share |
| ----------------- | ---------: | ----: |
| **APPROVE**       |     29,033 | 47.2% |
| **MANUAL_REVIEW** |     27,654 | 45.0% |
| **REJECT**        |      4,816 |  7.8% |

Average predicted default probability:

**8.04%**

---

## ⚡ FastAPI

The project exposes the credit-risk model through a REST API.

### Start API

```powershell
uvicorn src.api.main:app --reload
```

API URL:

```text
http://127.0.0.1:8000
```

Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

### Health Check

```text
GET /health
```

### Prediction Endpoint

```text
POST /predict
```

Example request:

```json
{
  "applicant_id": 396899
}
```

Example response:

```json
{
  "applicant_id": 396899,
  "default_probability": 0.06305131775793747,
  "risk_grade": "B",
  "decision": "MANUAL_REVIEW"
}
```

The endpoint also returns **SHAP-based positive and negative contributors**.

---

## 🖥️ Streamlit Dashboard

The project includes an interactive **Streamlit dashboard**.

Start it with:

```powershell
streamlit run src/dashboard/app.py
```

The dashboard displays:

* **Default probability**
* **Risk grade**
* **Business decision**
* **Risk interpretation**
* **Positive SHAP contributors**
* **Negative SHAP contributors**

The Streamlit application communicates with the **FastAPI backend**.

---

## 🐳 Docker

The FastAPI application is containerized using Docker.

### Build Image

```powershell
docker build -t credit-risk-api .
```

### Run Container

```powershell
docker run -d --name credit-risk-api-container -p 8000:8000 credit-risk-api
```

### Health Check

```powershell
curl http://localhost:8000/health
```

The Docker image packages the API, model, required data, and Python dependencies.

---

## 🧪 Testing

Automated tests cover:

* **Decision-engine boundaries**
* **Data validation**
* **Feature engineering**

Run:

```powershell
pytest -q
```

Current result:

```text
9 passed
```

Python syntax validation:

```powershell
python -m compileall src tests
```

---

## 📁 Project Structure

```text
credit-risk-intelligence1/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── 01_initial_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_shap_visualization.ipynb
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── dashboard/
│   │   └── app.py
│   │
│   ├── data/
│   │   ├── data_loader.py
│   │   └── data_validation.py
│   │
│   ├── features/
│   │   ├── data_preparation.py
│   │   ├── feature_engineering.py
│   │   └── preprocessing.py
│   │
│   └── models/
│       ├── baseline_model.py
│       ├── random_forest_model.py
│       ├── xgboost_model.py
│       ├── lightgbm_model.py
│       ├── model_evaluation.py
│       ├── threshold_analysis.py
│       ├── calibration.py
│       ├── model_selection.py
│       ├── cross_validation.py
│       ├── shap_explainability.py
│       ├── decision_engine.py
│       ├── portfolio_analysis.py
│       └── credit_risk_engine.py
│
├── tests/
├── models/
├── docs/
├── Dockerfile
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## 🛠️ Engineering Practices

This project demonstrates practical ML engineering concepts:

* **Stratified train/validation splitting**
* **Leakage-safe preprocessing**
* **Reusable scikit-learn pipelines**
* **Class-imbalance awareness**
* **Multiple model benchmarking**
* **Cross-validation**
* **Probability calibration**
* **Threshold analysis**
* **SHAP explainability**
* **Modular Python architecture**
* **REST API development**
* **Interactive dashboard development**
* **Automated testing**
* **Docker containerization**
* **Git/GitHub version control**

---

## ⚠️ Limitations

This is a **portfolio project**, not a production lending system.

Important limitations:

* The model primarily uses the Home Credit application table.
* Decision thresholds are illustrative.
* Model performance may change on new populations.
* SHAP explains model behavior rather than causality.
* Production lending requires fairness and bias analysis.
* Production systems require regulatory compliance and governance.
* Human oversight should remain part of high-impact credit decisions.

---

## 🔮 Future Improvements

Potential improvements include:

* **Integrate additional Home Credit tables**
* **Hyperparameter optimization**
* **Advanced probability calibration**
* **Cost-sensitive threshold optimization**
* **Fairness and bias analysis**
* **Model monitoring**
* **Data drift detection**
* **PostgreSQL integration**
* **API authentication**
* **CI/CD pipeline**
* **Cloud deployment**

---

## 👨‍💻 Author

**Wasique**

Machine Learning / Data Science Portfolio Project

### GitHub

**Repository:** `wasique-19/credit-risk-intelligence1`
