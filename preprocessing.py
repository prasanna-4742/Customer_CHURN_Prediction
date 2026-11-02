"""Data cleaning, feature engineering and the sklearn preprocessing pipeline."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

TARGET = "Churn"
ID_COL = "customerID"

SERVICE_COLS = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
SUPPORT_COLS = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
STREAMING_COLS = ["StreamingTV", "StreamingMovies"]

RAW_CATEGORICAL = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                   "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                   "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                   "PaperlessBilling", "PaymentMethod"]
RAW_NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_COLUMNS = RAW_CATEGORICAL + RAW_NUMERIC

ENGINEERED_NUMERIC = ["avg_monthly_spend", "num_services", "support_services",
                      "streaming_services", "has_internet", "is_auto_pay"]
ENGINEERED_CATEGORICAL = ["tenure_group"]

NUMERIC_FEATURES = RAW_NUMERIC + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = RAW_CATEGORICAL + ENGINEERED_CATEGORICAL


def _tidy_text(value):
    """Strip whitespace; blank strings become NaN."""
    if isinstance(value, str):
        value = value.strip()
        return value if value else np.nan
    return value


# ---------------------------------------------------------------- cleaning
def clean_data(df):
    """Handle missing / invalid values and normalise text. Returns a new DataFrame."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # text columns: strip whitespace, blanks -> NaN (works on pandas 2 and 3)
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype(object).map(_tidy_text)

    # numeric columns: coerce invalid strings (e.g. " ") to NaN
    for col in RAW_NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # invalid (negative) numeric values -> NaN
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    # SeniorCitizen must be 0/1
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].where(df["SeniorCitizen"].isin([0, 1]))

    # Fill TotalCharges when it can be derived (tenure * MonthlyCharges)
    if {"TotalCharges", "tenure", "MonthlyCharges"} <= set(df.columns):
        df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])

    # "No internet service" / "No phone service" are just "No" for modelling
    for col in SERVICE_COLS:
        if col in df.columns:
            df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})
    return df


def clean_target(df):
    """Map Churn Yes/No -> 1/0 and drop rows with unusable target."""
    df = df.copy()
    df[TARGET] = (df[TARGET].astype(object).map(_tidy_text).astype(str).str.lower()
                  .map({"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}))
    return df.dropna(subset=[TARGET]).astype({TARGET: int})


# ------------------------------------------------------ feature engineering
def engineer_features(df):
    """Create tenure category, average monthly spend and service usage indicators."""
    df = df.copy()
    for col in RAW_COLUMNS:           # make sure every expected column exists
        if col not in df.columns:
            df[col] = np.nan

    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    monthly = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    total = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # 1) customer tenure category
    group = pd.cut(tenure, bins=[-0.1, 12, 24, 48, np.inf],
                   labels=["0-1yr", "1-2yr", "2-4yr", "4yr+"]).astype(object)
    df["tenure_group"] = group.where(group.notna(), np.nan)

    # 2) average monthly spending; brand-new customers fall back to MonthlyCharges
    df["avg_monthly_spend"] = (total / tenure.where(tenure > 0)).fillna(monthly)

    # 3) service usage indicators
    def is_yes(col):
        return (df[col] == "Yes").astype(int)

    df["num_services"] = sum(is_yes(c) for c in SERVICE_COLS)
    df["support_services"] = sum(is_yes(c) for c in SUPPORT_COLS)
    df["streaming_services"] = sum(is_yes(c) for c in STREAMING_COLS)
    df["has_internet"] = df["InternetService"].isin(["DSL", "Fiber optic"]).astype(int)
    df["is_auto_pay"] = df["PaymentMethod"].astype(str).str.contains(
        "automatic", case=False).astype(int)
    return df


def prepare_frame(df):
    """Full raw -> model-ready frame (clean + engineer). Used in training and serving."""
    return engineer_features(clean_data(df))


# --------------------------------------------------------------- pipeline
def build_preprocessor():
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    column_tf = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES)])
    return Pipeline([("prepare", FunctionTransformer(prepare_frame, validate=False)),
                     ("columns", column_tf)])
