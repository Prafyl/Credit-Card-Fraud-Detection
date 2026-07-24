"""Loading and splitting the credit card dataset.

This is the only module that touches the raw CSV, so the de-duplication and the
train/validation/test split are defined exactly once and shared by eda.py and train.py.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "creditcard.csv"

# V1..V28 are anonymised PCA components; Amount is the only raw feature we keep.
# `Time` is deliberately excluded: it is "seconds since the first transaction in this
# 2-day sample", so it cannot generalise to a new transaction. EDA still uses it.
FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET = "Class"

RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    """Read the CSV and drop the 1,081 exact duplicate rows.

    Duplicates must go *before* splitting, otherwise identical rows land in both train
    and test and silently inflate every score.
    """
    return pd.read_csv(DATA_PATH).drop_duplicates()


def get_splits(df: pd.DataFrame | None = None):
    """Stratified 60/20/20 train/validation/test split.

    Three splits rather than two because the default 0.5 probability cut-off is a poor
    operating point at 0.167% positives. We fit on train, choose the decision threshold
    on validation, and report final numbers on test, which no decision ever touched.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    df = load_data() if df is None else df
    X, y = df[FEATURES], df[TARGET]

    # First carve off the test set, then split the remainder 75/25 -> 60/20 overall.
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = get_splits()
    for name, y in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"{name:5s}  rows={len(y):6,d}  frauds={y.sum():3d}  rate={y.mean():.4%}")
