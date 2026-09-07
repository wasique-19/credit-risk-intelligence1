import pandas as pd

from sklearn.model_selection import train_test_split

from src.features.feature_engineering import create_features
from src.features.preprocessing import build_preprocessor


def prepare_train_validation_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
):
    """Create features, split data, and build a leakage-safe preprocessor."""

    df = create_features(df)

    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    return X_train, X_valid, y_train, y_valid, preprocessor