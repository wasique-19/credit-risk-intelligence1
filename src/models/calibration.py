from sklearn.calibration import CalibratedClassifierCV


def calibrate_model(model, X_train, y_train, method="sigmoid"):
    """Calibrate a fitted classification model."""

    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv=3,
    )

    calibrated_model.fit(X_train, y_train)

    return calibrated_model