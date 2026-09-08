from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_logistic_regression_model(preprocessor):
    """Build a preprocessing + Logistic Regression pipeline."""

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    return model

from pathlib import Path
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


def save_model(model, filename="logistic_regression_baseline.joblib"):
    """Save a trained model to the models directory."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / filename
    joblib.dump(model, model_path)

    return model_path


def load_model(filename="logistic_regression_baseline.joblib"):
    """Load a saved model from the models directory."""

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)