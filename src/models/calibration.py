from sklearn.calibration import CalibratedClassifierCV
from pathlib import Path

import joblib
from sklearn.calibration import CalibratedClassifierCV


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

def calibrate_model(model, X_train, y_train, method="sigmoid"):
    """Calibrate a fitted classification model."""

    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=3,
    )

    calibrated_model.fit(X_train, y_train)

    return calibrated_model

def load_calibrated_model(filename="lightgbm_calibrated.joblib"):
    """Load the saved calibrated model."""
    model_path = MODEL_DIR / filename
    if not model_path.exists():
        raise FileNotFoundError(f"Calibrated model not found: {model_path}")
    return joblib.load(model_path)