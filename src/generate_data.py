"""
Synthetic telecom-style customer dataset generator.

Features are behavioral (tenure, charges, contract, support contacts, usage),
and the churn label is generated from a known logistic function of a few
driver features so the "true" churn drivers are recoverable by the analysis.

Output: data/customer_churn.csv
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def generate(n_customers: int = 5_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 73, n_customers)
    monthly_charges = np.round(rng.normal(65, 25, n_customers).clip(15, 150), 2)
    total_charges = np.round(monthly_charges * tenure_months + rng.normal(0, 40, n_customers), 2)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], n_customers, p=[0.55, 0.25, 0.20]
    )
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        n_customers, p=[0.34, 0.20, 0.23, 0.23],
    )
    internet = rng.choice(["DSL", "Fiber optic", "No"], n_customers, p=[0.30, 0.45, 0.25])
    support_calls = rng.poisson(1.2, n_customers).clip(0, 12)
    usage_gb = np.round(rng.gamma(2.0, 25.0, n_customers), 1)
    late_payments = rng.poisson(0.5, n_customers).clip(0, 6)
    senior = rng.binomial(1, 0.16, n_customers)
    dependents = rng.binomial(1, 0.30, n_customers)

    # Ground-truth churn logit: known drivers
    logit = (
        -2.2
        + 0.85 * (contract == "Month-to-month")
        - 0.45 * (contract == "Two year")
        + 0.030 * monthly_charges
        + 0.55 * np.log1p(support_calls)
        + 0.35 * late_payments
        - 0.035 * tenure_months
        + 0.30 * (internet == "Fiber optic")
        - 0.25 * dependents
        + rng.normal(0, 0.35, n_customers)
    )
    proba = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n_customers) < proba).astype(int)

    return pd.DataFrame(
        {
            "customer_id": [f"C{10_000 + i}" for i in range(n_customers)],
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": total_charges,
            "contract": contract,
            "payment_method": payment,
            "internet_service": internet,
            "support_calls": support_calls,
            "usage_gb": usage_gb,
            "late_payments": late_payments,
            "senior_citizen": senior,
            "has_dependents": dependents,
            "churn": churn,
        }
    )


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = generate()
    path = DATA_DIR / "customer_churn.csv"
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} customers (churn rate {df['churn'].mean():.1%}) to {path}")


if __name__ == "__main__":
    main()
