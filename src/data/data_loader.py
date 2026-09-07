from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_application_train() -> pd.DataFrame:
    """Load the Home Credit application training dataset."""
    file_path = RAW_DATA_DIR / "application_train.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Training file not found: {file_path}")

    return pd.read_csv(file_path)


def load_application_test() -> pd.DataFrame:
    """Load the Home Credit application test dataset."""
    file_path = RAW_DATA_DIR / "application_test.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"Test file not found: {file_path}")

    return pd.read_csv(file_path)