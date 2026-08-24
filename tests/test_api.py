import pytest
from src.app import app

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json["status"] == "healthy"

def test_predict_route(client):
    response = client.post("/predict", json={"simulate": True})
    assert response.status_code == 200
    assert "prediction" in response.json
    assert "confidence_score" in response.json