import numpy as np
import pandas as pd


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features for the Home Credit application data."""

    df = df.copy()

    # Financial ratios
    df["LOAN_INCOME_RATIO"] = (
        df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    )

    df["ANNUITY_INCOME_RATIO"] = (
        df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    )

    df["CREDIT_GOODS_RATIO"] = (
        df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    )

    # Age and employment
    df["AGE_YEARS"] = (
        -df["DAYS_BIRTH"] / 365.25
    )

    df["DAYS_EMPLOYED_CLEAN"] = (
        df["DAYS_EMPLOYED"].replace(365243, np.nan)
    )

    df["EMPLOYMENT_YEARS"] = (
        -df["DAYS_EMPLOYED_CLEAN"] / 365.25
    )

    # Household and income features
    df["INCOME_PER_CHILD"] = (
        df["AMT_INCOME_TOTAL"] / (df["CNT_CHILDREN"] + 1)
    )

    df["INCOME_PER_FAMILY_MEMBER"] = (
        df["AMT_INCOME_TOTAL"] / (df["CNT_FAM_MEMBERS"] + 1)
    )

    # Credit burden
    df["CREDIT_MINUS_INCOME"] = (
        df["AMT_CREDIT"] - df["AMT_INCOME_TOTAL"]
    )

    df["ANNUITY_MINUS_INCOME"] = (
        df["AMT_ANNUITY"] - df["AMT_INCOME_TOTAL"]
    )

    # Employment relative to age
    df["EMPLOYMENT_AGE_RATIO"] = (
        df["EMPLOYMENT_YEARS"] / df["AGE_YEARS"]
    )

    return df