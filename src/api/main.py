from src.models.decision_engine import make_credit_decision
from src.data.data_loader import load_application_train
from src.features.feature_engineering import create_features
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "lightgbm_calibrated.joblib"

model = joblib.load(MODEL_PATH)

application_data = load_application_train()
application_data = create_features(application_data)

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

    return {
        "applicant_id": request.applicant_id,
        "default_probability": probability,
        "risk_grade": decision["risk_grade"],
        "decision": decision["decision"],
    }