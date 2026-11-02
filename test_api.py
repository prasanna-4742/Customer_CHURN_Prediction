from fastapi.testclient import TestClient

from app.main import app

PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
    "tenure": 3, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.5, "TotalCharges": 286.5}


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_valid():
    with TestClient(app) as client:
        r = client.post("/predict", json=PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert body["prediction"] in ("Churn", "Not Churn")
        assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_without_total_charges():
    payload = {k: v for k, v in PAYLOAD.items() if k != "TotalCharges"}
    with TestClient(app) as client:
        assert client.post("/predict", json=payload).status_code == 200


def test_predict_invalid_payload():
    with TestClient(app) as client:
        assert client.post("/predict", json={"gender": "Male"}).status_code == 422
