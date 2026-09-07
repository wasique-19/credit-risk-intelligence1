import pandas as pd


def validate_application_data(
    df: pd.DataFrame,
    is_train: bool = True,
) -> None:
    """Validate the basic structure of an application dataset."""

    if "SK_ID_CURR" not in df.columns:
        raise ValueError("Missing required column: SK_ID_CURR")

    if is_train and "TARGET" not in df.columns:
        raise ValueError("Training data must contain TARGET")

    if not is_train and "TARGET" in df.columns:
        raise ValueError("Test data must not contain TARGET")

    if df["SK_ID_CURR"].duplicated().any():
        raise ValueError("Duplicate SK_ID_CURR values found")

    print("Data validation passed.")