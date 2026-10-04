from fastapi.testclient import TestClient

from app.main import app
from src.predict import predict_one

HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 75.0, "TotalCharges": 75.0,
}

LOW_RISK = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 60, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Two year",
    "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)",
    "MonthlyCharges": 55.0, "TotalCharges": 3300.0,
}


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_high_risk_customer_is_flagged():
    with TestClient(app) as client:
        r = client.post("/predict", json=HIGH_RISK)
        assert r.status_code == 200
        body = r.json()
        assert body["prediction"] == "Churn"
        assert 0 <= body["churn_probability"] <= 1


def test_low_risk_customer_is_not_flagged():
    with TestClient(app) as client:
        r = client.post("/predict", json=LOW_RISK)
        assert r.status_code == 200
        assert r.json()["prediction"] == "No Churn"


def test_api_matches_direct_prediction():
    with TestClient(app) as client:
        api = client.post("/predict", json=HIGH_RISK).json()
    direct = predict_one(HIGH_RISK)
    assert api["churn_probability"] == direct["churn_probability"]


def test_missing_total_charges_is_estimated():
    payload = {k: v for k, v in HIGH_RISK.items() if k != "TotalCharges"}
    with TestClient(app) as client:
        r = client.post("/predict", json=payload)
        assert r.status_code == 200
        assert r.json()["total_charges_estimated"] is True


def test_invalid_category_is_rejected():
    bad = {**HIGH_RISK, "Contract": "month to month"}
    with TestClient(app) as client:
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


def test_missing_field_is_rejected():
    bad = {k: v for k, v in HIGH_RISK.items() if k != "tenure"}
    with TestClient(app) as client:
        r = client.post("/predict", json=bad)
        assert r.status_code == 422