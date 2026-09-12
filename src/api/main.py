from src.models.decision_engine import make_credit_decision
from src.data.data_loader import load_application_train
from src.features.feature_engineering import create_features
from src.models.shap_explainability import explain_prediction
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib
import shap
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_calibrated.joblib"

model = joblib.load(MODEL_PATH)

lgbm = model.estimator.named_steps["classifier"]
shap_explainer = shap.TreeExplainer(lgbm)

application_data = load_application_train()
application_data = create_features(application_data)

api_preprocessor = model.estimator.named_steps["preprocessor"]
feature_names = api_preprocessor.get_feature_names_out()

class PredictionRequest(BaseModel):
    applicant_id: int

app = FastAPI(
    title="Credit Risk Intelligence API",
    description="API for credit default risk prediction.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "Credit Risk Intelligence API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }

@app.post("/predict")
def predict(request: PredictionRequest):
    row = application_data[
        application_data["SK_ID_CURR"] == request.applicant_id
    ]

    if row.empty:
        return {
            "error": "Applicant not found",
            "applicant_id": request.applicant_id,
        }

    X = row.drop(columns=["TARGET"])

    probability = float(model.predict_proba(X)[:, 1][0])

    decision = make_credit_decision(probability)

    X_transformed = api_preprocessor.transform(X)

    shap_values = shap_explainer.shap_values(X_transformed)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    explanation = explain_prediction(
        shap_values,
        feature_names,
        row_index=0,
        top_n=5,
    )

    return {
        "applicant_id": request.applicant_id,
        "default_probability": probability,
        "risk_grade": decision["risk_grade"],
        "decision": decision["decision"],
        "positive_contributors": [
            {
                "feature": feature,
                "shap_value": float(value),
            }
            for feature, value in explanation["positive"]
        ],
        "negative_contributors": [
            {
                "feature": feature,
                "shap_value": float(value),
            }
            for feature, value in explanation["negative"]
        ],
    }