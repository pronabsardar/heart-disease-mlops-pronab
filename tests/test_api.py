"""Tests for FastAPI application."""
import sys
import os
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def client():
    if not os.path.exists("models/best_model.pkl"):
        pytest.skip("Model not trained yet")
    from src.api import app
    return TestClient(app)


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["author"] == "Pronab Sardar"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict(client):
    payload = {
        "age": 55, "sex": 1, "cp": 3, "trestbps": 140, "chol": 250,
        "fbs": 0, "restecg": 1, "thalach": 150, "exang": 0,
        "oldpeak": 1.5, "slope": 2, "ca": 0, "thal": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0 <= data["confidence"] <= 1
    assert data["risk_label"] in ["High Risk", "Low Risk"]


def test_predict_invalid_input(client):
    payload = {"age": 55}  # Missing fields
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_metrics_endpoint(client):
    response = client.get("/metrics")
    assert response.status_code == 200