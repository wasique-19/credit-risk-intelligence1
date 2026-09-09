from pathlib import Path

import joblib
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


def build_lightgbm_model(preprocessor):
    """Build a preprocessing + LightGBM pipeline."""

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LGBMClassifier(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="binary",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )

    return model


def save_lightgbm_model(
    model,
    filename="lightgbm_baseline.joblib",
):
    """Save the trained LightGBM model."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / filename
    joblib.dump(model, model_path)

    return model_path


def load_lightgbm_model(
    filename="lightgbm_baseline.joblib",
):
    """Load the saved LightGBM model."""

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)