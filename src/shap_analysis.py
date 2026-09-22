"""
SHAP driver analysis for the best churn model.

- Loads models/best_model.joblib and a sample of the training data.
- Computes SHAP values (TreeExplainer when the best model is tree-based and
  shap is installed; otherwise falls back to permutation importance).
- Saves a SHAP beeswarm summary plot to images/shap_summary.png
  (or a bar plot of mean |SHAP| / permutation importance).
- Prints the top churn drivers to stdout.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

try:
    import shap

    HAS_SHAP = True
except ImportError:  # pragma: no cover
    HAS_SHAP = False

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "customer_churn.csv"
MODEL_DIR = BASE_DIR / "models"
IMAGE_DIR = BASE_DIR / "images"

NUMERIC = [
    "tenure_months", "monthly_charges", "total_charges", "support_calls",
    "usage_gb", "late_payments", "senior_citizen", "has_dependents",
]
CATEGORICAL = ["contract", "payment_method", "internet_service"]
FEATURES = NUMERIC + CATEGORICAL
TOP_N = 10


def clean_name(name: str) -> str:
    return name.replace("num__", "").replace("cat__", "").replace("_", " ")


def main(n_sample: int = 500, seed: int = 42) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    pipe = joblib.load(MODEL_DIR / "best_model.joblib")
    feature_names = pd.read_csv(MODEL_DIR / "feature_names.csv")["0"].tolist()

    sample = df.sample(n=min(n_sample, len(df)), random_state=seed)
    X_raw = sample[FEATURES]
    X_proc = pipe.named_steps["prep"].transform(X_raw)
    clf = pipe.named_steps["clf"]
    clf_name = type(clf).__name__

    use_tree_shap = HAS_SHAP and clf_name in {"XGBClassifier", "RandomForestClassifier"}
    use_linear_shap = HAS_SHAP and clf_name == "LogisticRegression"
    if use_tree_shap or use_linear_shap:
        if use_tree_shap:
            explainer = shap.TreeExplainer(clf)
        else:
            explainer = shap.LinearExplainer(clf, X_proc)
        shap_values = explainer.shap_values(X_proc)
        if isinstance(shap_values, list):  # binary classification -> pick class 1
            shap_values = shap_values[1]
        mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)

        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(9, 6))
        shap.summary_plot(shap_values, X_proc, feature_names=feature_names,
                          show=False, max_display=TOP_N)
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / "shap_summary.png", dpi=150)
        plt.close()
        print(f"SHAP summary plot saved to {IMAGE_DIR / 'shap_summary.png'}")
    else:
        if not HAS_SHAP:
            print("Note: shap not installed — using permutation importance fallback.")
        r = permutation_importance(
            pipe, X_raw, sample["churn"], n_repeats=10, random_state=seed, n_jobs=-1
        )
        mean_abs = r.importances_mean
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        order = np.argsort(mean_abs)[::-1][:TOP_N]
        plt.figure(figsize=(9, 6))
        plt.barh([clean_name(feature_names[i]) for i in order][::-1],
                 mean_abs[order][::-1])
        plt.xlabel("Permutation importance (mean decrease in score)")
        plt.title("Top churn drivers (permutation importance)")
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / "shap_summary.png", dpi=150)
        plt.close()

    importance = (
        pd.DataFrame(
            {"feature": [clean_name(f) for f in feature_names],
             "mean_abs_shap": mean_abs}
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance.to_csv(MODEL_DIR / "feature_importance.csv", index=False)

    print("\nTop churn drivers:")
    for i, row in importance.head(TOP_N).iterrows():
        print(f"  {i + 1:2d}. {row['feature']:35s} {row['mean_abs_shap']:.4f}")
    return importance


if __name__ == "__main__":
    main()
