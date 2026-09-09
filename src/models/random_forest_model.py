from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"


def build_random_forest_model(preprocessor):
    """Build a preprocessing + Random Forest pipeline."""

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


def save_random_forest_model(
    model,
    filename="random_forest_baseline.joblib",
):
    """Save the trained Random Forest model."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / filename
    joblib.dump(model, model_path)

    return model_path


def load_random_forest_model(
    filename="random_forest_baseline.joblib",
):
    """Load the saved Random Forest model."""

    model_path = MODEL_DIR / filename

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    return joblib.load(model_path)