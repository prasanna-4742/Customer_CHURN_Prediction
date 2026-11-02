"""Generate a synthetic Telco-style churn dataset (same schema as the IBM/Kaggle
'WA_Fn-UseC_-Telco-Customer-Churn.csv').

Use this ONLY for practice / CI. For the exam, copy the examiner's CSV to
data/telco_churn.csv (same column names) and re-run:  python -m src.train
"""
import argparse

import numpy as np
import pandas as pd


def generate(n=3000, seed=42):
    rng = np.random.default_rng(seed)
    yn = ["Yes", "No"]
    tenure = np.clip(rng.gamma(2.0, 18.0, n).astype(int), 0, 72)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    phone = rng.choice(yn, n, p=[0.9, 0.1])

    def addon(p=0.4):
        vals = rng.choice(yn, n, p=[p, 1 - p]).astype(object)
        vals[internet == "No"] = "No internet service"
        return vals

    df = pd.DataFrame({
        "customerID": [f"{rng.integers(1000, 9999)}-{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(yn, n),
        "Dependents": rng.choice(yn, n, p=[0.3, 0.7]),
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": np.where(phone == "No", "No phone service", rng.choice(yn, n)),
        "InternetService": internet,
        "OnlineSecurity": addon(0.3),
        "OnlineBackup": addon(0.35),
        "DeviceProtection": addon(0.35),
        "TechSupport": addon(0.3),
        "StreamingTV": addon(0.45),
        "StreamingMovies": addon(0.45),
        "Contract": contract,
        "PaperlessBilling": rng.choice(yn, n, p=[0.6, 0.4]),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"], n, p=[0.34, 0.23, 0.22, 0.21]),
    })
    base = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 45, 20))
    extras = sum((df[c] == "Yes").astype(int) * 5 for c in
                 ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
                  "StreamingTV", "StreamingMovies"])
    monthly = np.round(base + extras + (phone == "Yes") * 5 + rng.normal(0, 3, n), 2)
    df["MonthlyCharges"] = np.clip(monthly, 18.25, 118.75)
    df["TotalCharges"] = np.round(df["MonthlyCharges"] * df["tenure"]
                                  + rng.normal(0, 20, n).clip(-30, 30) * (tenure > 0), 2)
    df["TotalCharges"] = df["TotalCharges"].clip(lower=0)

    logit = (-1.0
             + 1.3 * (contract == "Month-to-month") - 1.2 * (contract == "Two year")
             + 0.7 * (internet == "Fiber optic")
             - 0.04 * tenure
             + 0.012 * (df["MonthlyCharges"] - 65)
             + 0.5 * (df["PaymentMethod"] == "Electronic check")
             + 0.3 * df["SeniorCitizen"]
             - 0.5 * (df["TechSupport"] == "Yes") - 0.4 * (df["OnlineSecurity"] == "Yes"))
    prob = 1 / (1 + np.exp(-logit))
    df["Churn"] = np.where(rng.random(n) < prob, "Yes", "No")

    # Realistic dirty data: blank TotalCharges (as in the real dataset), NaNs, invalid values
    df["TotalCharges"] = df["TotalCharges"].astype(object)
    df.loc[df["tenure"] == 0, "TotalCharges"] = " "
    df.loc[rng.choice(n, 15, replace=False), "TotalCharges"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "MonthlyCharges"] = np.nan
    df.loc[rng.choice(n, 5, replace=False), "tenure"] = -1
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=3000)
    ap.add_argument("--out", default="data/telco_churn.csv")
    a = ap.parse_args()
    generate(a.rows).to_csv(a.out, index=False)
    print(f"Wrote {a.rows} rows to {a.out}")
