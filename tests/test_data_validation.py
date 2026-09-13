import pandas as pd
import pytest

from src.data.data_validation import validate_application_data


def test_valid_training_data():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "TARGET": [0, 1, 0],
    })

    validate_application_data(df, is_train=True)


def test_missing_id_column():
    df = pd.DataFrame({
        "TARGET": [0, 1],
    })

    with pytest.raises(ValueError, match="SK_ID_CURR"):
        validate_application_data(df, is_train=True)


def test_duplicate_id():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 1, 2],
        "TARGET": [0, 1, 0],
    })

    with pytest.raises(ValueError, match="Duplicate SK_ID_CURR"):
        validate_application_data(df, is_train=True)


def test_test_data_must_not_have_target():
    df = pd.DataFrame({
        "SK_ID_CURR": [1, 2, 3],
        "TARGET": [0, 1, 0],
    })

    with pytest.raises(ValueError, match="must not contain TARGET"):
        validate_application_data(df, is_train=False)