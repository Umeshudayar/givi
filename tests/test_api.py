import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_home_page():
    response = client.get("/")
    assert response.status_code == 200

def test_predict_endpoint():
    test_data = {
        "restaurant": "Test Restaurant",
        "location": "Test Location",
        "orderTime": "12:30",
        "orderType": "standard"
    }
    response = client.post("/api/predict", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "estimatedTime" in data
    assert "confidence" in data
    assert "factors" in data
    assert isinstance(data["estimatedTime"], int)
    assert 0 <= data["confidence"] <= 1
