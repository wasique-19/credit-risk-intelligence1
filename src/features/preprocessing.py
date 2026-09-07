import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing pipeline for numeric and categorical features."""

    numeric_features = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    return preprocessor


def prepare_data(
    df: pd.DataFrame,
    target_column: str = "TARGET",
):
    """Separate target from features and build the preprocessing pipeline."""

    if target_column not in df.columns:
        raise ValueError(f"Target column not found: {target_column}")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    preprocessor = build_preprocessor(X)

    return X, y, preprocessor