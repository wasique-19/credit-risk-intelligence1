from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


def build_xgboost_model(preprocessor):
    """Build a preprocessing + XGBoost pipeline."""

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                XGBClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary:logistic",
                    eval_metric="auc",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


def save_xgboost_model(
    model,
    filename="xgboost_baseline.joblib",
):
    """Save the trained XGBoost model."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / filename
    joblib.dump(model, model_path)

    return model_path


def load_xgboost_model(
    filename="xgboost_baseline.joblib",
):
    """Load the saved XGBoost model."""

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)