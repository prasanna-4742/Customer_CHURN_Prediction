import numpy as np
import pandas as pd

from src.preprocessing import clean_data, clean_target, engineer_features


def sample():
    return pd.DataFrame([{
        "customerID": "A-1", "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes",
        "Dependents": "No", "tenure": 0, "PhoneService": "No",
        "MultipleLines": "No phone service", "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "No", "DeviceProtection": "No",
        "TechSupport": "Yes", "StreamingTV": "No", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 50.0,
        "TotalCharges": " ", "Churn": "Yes"},
        {"customerID": "A-2", "gender": "Female", "SeniorCitizen": 1, "Partner": "No",
         "Dependents": "No", "tenure": -3, "PhoneService": "Yes", "MultipleLines": "Yes",
         "InternetService": "No", "OnlineSecurity": "No internet service",
         "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
         "TechSupport": "No internet service", "StreamingTV": "No internet service",
         "StreamingMovies": "No internet service", "Contract": "Two year",
         "PaperlessBilling": "No", "PaymentMethod": "Mailed check",
         "MonthlyCharges": 20.0, "TotalCharges": "240", "Churn": "No"}])


def test_blank_total_charges_is_filled():
    out = clean_data(sample())
    assert not out["TotalCharges"].isna().iloc[0]


def test_negative_tenure_becomes_nan():
    assert np.isnan(clean_data(sample())["tenure"].iloc[1])


def test_no_internet_service_is_normalised():
    assert clean_data(sample())["OnlineSecurity"].iloc[1] == "No"


def test_target_encoding():
    assert clean_target(sample())["Churn"].tolist() == [1, 0]


def test_engineered_features():
    out = engineer_features(clean_data(sample()))
    for col in ["tenure_group", "avg_monthly_spend", "num_services",
                "support_services", "streaming_services", "has_internet", "is_auto_pay"]:
        assert col in out.columns
    assert out["num_services"].iloc[0] == 3        # OnlineSecurity, TechSupport, StreamingMovies
    assert out["avg_monthly_spend"].iloc[0] == 50.0  # tenure 0 -> falls back to MonthlyCharges
    assert out["is_auto_pay"].iloc[0] == 1
    assert out["has_internet"].iloc[1] == 0
