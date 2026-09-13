import pandas as pd

from src.features.feature_engineering import create_features


def test_create_features_adds_expected_columns():
    df = pd.DataFrame({
        "AMT_CREDIT": [100000],
        "AMT_INCOME_TOTAL": [50000],
        "AMT_ANNUITY": [5000],
        "AMT_GOODS_PRICE": [90000],
        "DAYS_BIRTH": [-10000],
        "DAYS_EMPLOYED": [-2000],
        "CNT_CHILDREN": [1],
        "CNT_FAM_MEMBERS": [3],
    })

    result = create_features(df)

    expected_columns = [
        "LOAN_INCOME_RATIO",
        "ANNUITY_INCOME_RATIO",
        "CREDIT_GOODS_RATIO",
        "AGE_YEARS",
        "DAYS_EMPLOYED_CLEAN",
        "EMPLOYMENT_YEARS",
        "INCOME_PER_CHILD",
        "INCOME_PER_FAMILY_MEMBER",
        "CREDIT_MINUS_INCOME",
        "ANNUITY_MINUS_INCOME",
        "EMPLOYMENT_AGE_RATIO",
    ]

    for column in expected_columns:
        assert column in result.columns


def test_create_features_calculates_ratios():
    df = pd.DataFrame({
        "AMT_CREDIT": [100000],
        "AMT_INCOME_TOTAL": [50000],
        "AMT_ANNUITY": [5000],
        "AMT_GOODS_PRICE": [90000],
        "DAYS_BIRTH": [-10000],
        "DAYS_EMPLOYED": [-2000],
        "CNT_CHILDREN": [1],
        "CNT_FAM_MEMBERS": [3],
    })

    result = create_features(df)

    assert result.loc[0, "LOAN_INCOME_RATIO"] == 2.0
    assert result.loc[0, "ANNUITY_INCOME_RATIO"] == 0.1
    assert result.loc[0, "CREDIT_GOODS_RATIO"] == 100000 / 90000