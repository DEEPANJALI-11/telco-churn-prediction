"""
Train the final churn pipeline, choose the threshold on training CV,
evaluate once on the test set, and save one artifact.

Run from the Telco folder:
    python -m src.train
    python -m src.train path/to/telco_clean.csv
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.preprocessing import (
    DEFAULT_DATA_PATH, RANDOM_STATE, ROOT,
    build_preprocessor, get_cat_cols, load_data, split_data,
)

MODEL_PATH = ROOT / "model" / "churn_xgb.joblib"

# Best hyperparameters from RandomizedSearchCV (CV ROC-AUC 0.8502)
BEST_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.5,
    reg_lambda=5,
)


def build_pipeline(cat_cols, scale_pos_weight):
    clf = XGBClassifier(
        **BEST_PARAMS,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return Pipeline([("pre", build_preprocessor(cat_cols)), ("clf", clf)])


def main():
    data_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_PATH
    X, y = load_data(data_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    cat_cols = get_cat_cols(X)

    spw = (y_train == 0).sum() / (y_train == 1).sum()
    pipe = build_pipeline(cat_cols, spw)

    # Threshold from out-of-fold predictions on TRAIN only (best F1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_proba = cross_val_predict(pipe, X_train, y_train, cv=cv,
                                 method="predict_proba")[:, 1]
    prec, rec, thr = precision_recall_curve(y_train, cv_proba)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-9)
    threshold = round(float(thr[np.argmax(f1)]), 3)
    print(f"Chosen threshold (best F1 on train CV): {threshold}")
    print(f"CV ROC-AUC: {roc_auc_score(y_train, cv_proba):.4f}")

    # Fit on the full training set
    pipe.fit(X_train, y_train)

    # Evaluate ONCE on the test set
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    }
    print("\nTest-set results")
    for k, v in metrics.items():
        print(f"  {k:<10}{v:.4f}")
    print("\nConfusion matrix (rows = actual, cols = predicted):")
    print(pd.DataFrame(confusion_matrix(y_test, pred),
                       index=["actual No", "actual Yes"],
                       columns=["pred No", "pred Yes"]))

    # Save pipeline + threshold + metadata together
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": pipe,
        "threshold": threshold,
        "columns": list(X.columns),
        "categorical_values": {c: sorted(X[c].unique().tolist()) for c in cat_cols},
        "test_metrics": metrics,
        "versions": {
            "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"\nSaved artifact to {MODEL_PATH}")
    print("Versions:", artifact["versions"])


if __name__ == "__main__":
    main()