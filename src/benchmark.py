"""
Benchmark churn classifiers with cross-validation.

Models:
  - LogisticRegression (scaled numeric + one-hot categoricals)
  - RandomForestClassifier
  - XGBoost (if installed; otherwise skipped with a note)

Metric: ROC-AUC via 5-fold stratified cross-validation.
Saves the best estimator (refit on full data) to models/best_model.joblib
and the CV summary to models/cv_results.csv.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
MODEL_DIR = BASE_DIR / "models"

NUMERIC = [
    "tenure_months", "monthly_charges", "total_charges", "support_calls",
    "usage_gb", "late_payments", "senior_citizen", "has_dependents",
]
CATEGORICAL = ["contract", "payment_method", "internet_service"]
FEATURES = NUMERIC + CATEGORICAL


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )


def candidate_models() -> dict:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            n_jobs=-1, random_state=42,
        ),
    }
    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            n_jobs=-1, random_state=42,
        )
    return models


def main() -> dict:
    df = pd.read_csv(DATA_PATH)
    X, y = df[FEATURES], df["churn"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results, fitted = [], {}
    for name, clf in candidate_models().items():
        pipe = Pipeline([("prep", make_preprocessor()), ("clf", clf)])
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
        results.append(
            {"model": name, "cv_auc_mean": scores.mean(), "cv_auc_std": scores.std()}
        )
        pipe.fit(X, y)
        fitted[name] = pipe
        print(f"{name:20s} ROC-AUC = {scores.mean():.4f} ± {scores.std():.4f}")

    if not HAS_XGBOOST:
        print("Note: xgboost not installed — XGBoost skipped (install it to include).")

    results_df = pd.DataFrame(results).sort_values("cv_auc_mean", ascending=False)
    best_name = results_df.iloc[0]["model"]
    print(f"\nBest model: {best_name} (ROC-AUC {results_df.iloc[0]['cv_auc_mean']:.4f})")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], MODEL_DIR / "best_model.joblib")
    results_df.to_csv(MODEL_DIR / "cv_results.csv", index=False)
    feature_names = fitted[best_name].named_steps["prep"].get_feature_names_out()
    pd.Series(list(feature_names)).to_csv(MODEL_DIR / "feature_names.csv", index=False)
    print(f"Saved best model ({best_name}) and CV summary to {MODEL_DIR}")
    return {"best_model": best_name, "results": results_df.to_dict("records")}


if __name__ == "__main__":
    main()
