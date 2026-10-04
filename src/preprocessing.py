"""Data loading, train/test split, and the preprocessing pipeline."""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT / "data" / "telco_clean.csv"

TARGET = "Churn"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
RANDOM_STATE = 42


def load_data(path=DEFAULT_DATA_PATH):
    """Load the cleaned CSV and return features X and binary target y."""
    df = pd.read_csv(path)
    assert df["TotalCharges"].dtype.kind == "f", "TotalCharges must be numeric"
    X = df.drop(columns=[TARGET])
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def split_data(X, y):
    """The one split used everywhere (train, evaluation, prediction checks)."""
    return train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)


def get_cat_cols(X):
    return [c for c in X.columns if c not in NUM_COLS]


def build_preprocessor(cat_cols):
    """Impute -> scale numerics, impute -> one-hot categoricals.
    Uses only built-in scikit-learn pieces, so the saved model has no
    dependency on this module."""
    return ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), NUM_COLS),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])