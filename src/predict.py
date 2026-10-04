"""
Load the saved artifact and predict churn for one raw customer.
FastAPI will import `predict_one` from here.

Self-check from the Telco folder:
    python -m src.predict
"""
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from src.preprocessing import ROOT, load_data, split_data

MODEL_PATH = ROOT / "model" / "churn_xgb.joblib"


@lru_cache(maxsize=1)
def load_artifact(path=MODEL_PATH):
    """Loaded once, then cached."""
    return joblib.load(path)


def predict_one(customer: dict) -> dict:
    """Raw customer dict in -> probability, label and threshold out."""
    art = load_artifact()

    missing = [c for c in art["columns"] if c not in customer]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    for col, allowed in art["categorical_values"].items():
        if customer[col] not in allowed:
            raise ValueError(f"Invalid value {customer[col]!r} for {col}. Allowed: {allowed}")

    row = pd.DataFrame([customer])[art["columns"]]    # enforce column order
    proba = float(art["model"].predict_proba(row)[0, 1])
    return {
        "churn_probability": round(proba, 4),
        "prediction": "Churn" if proba >= art["threshold"] else "No Churn",
        "threshold": art["threshold"],
    }


if __name__ == "__main__":
    art = load_artifact()
    print("Threshold:", art["threshold"])
    print("Saved test metrics:", {k: round(v, 4) for k, v in art["test_metrics"].items()})

    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    sample = list(y_test[y_test == 1].index[:3]) + list(y_test[y_test == 0].index[:3])

    print("\nRaw rows -> prediction")
    for idx in sample:
        out = predict_one(X_test.loc[idx].to_dict())
        actual = "Churn" if y_test.loc[idx] == 1 else "No Churn"
        print(f"  row {idx:>5} | actual: {actual:<8} | "
              f"prob: {out['churn_probability']:.3f} | predicted: {out['prediction']}")

    # Check 1: single-row path matches batch path
    batch = art["model"].predict_proba(X_test.loc[sample])[:, 1]
    single = np.array([predict_one(X_test.loc[i].to_dict())["churn_probability"] for i in sample])
    assert np.allclose(batch, single, atol=1e-4), "single-row and batch predictions differ"
    print("\nCheck 1 passed: single-row predictions match batch predictions.")

    # Check 2: reloaded model reproduces the saved test recall
    proba = art["model"].predict_proba(X_test)[:, 1]
    pred = (proba >= art["threshold"]).astype(int)
    recall = ((pred == 1) & (y_test == 1)).sum() / (y_test == 1).sum()
    assert abs(recall - art["test_metrics"]["recall"]) < 1e-6, "recall mismatch"
    print("Check 2 passed: reloaded model reproduces the saved test recall.")

    # Check 3: a blank numeric field doesn't crash (imputer handles it)
    c = X_test.loc[sample[0]].to_dict()
    c["TotalCharges"] = np.nan
    print("Check 3 passed: missing TotalCharges ->", predict_one(c))