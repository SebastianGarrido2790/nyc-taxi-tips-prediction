"""
Unit tests for the FastAPI serving layer (Predict API).
"""

from fastapi.testclient import TestClient
from src.api.predict_api import app

client = TestClient(app)


def test_health_check_endpoint():
    """Test the /health endpoint to ensure it returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_predict_endpoint_validation():
    """Test the /predict endpoint for validation errors when payload is missing required fields."""
    # Sending empty payload
    response = client.post("/predict", json=[{}])
    assert response.status_code == 422  # Unprocessable Entity

    # Missing required 'trip_distance'
    payload = [
        {
            "total_amount": 15.0,
            "passenger_count": 1,
            "ratecode_id": 1,
            "airport_fee": 0.0,
            "congestion_surcharge": 2.5,
            "tolls_amount": 0.0,
            "hour": 12,
            "day": 15,
            "month": 1,
        }
    ]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_predict_feature_importance_no_model():
    """
    Test feature importance endpoint behavior.
    Since the model isn't explicitly mocked to load here, it might return 503
    depending on whether lifespans are natively executed by TestClient.
    Using TestClient, lifespan events are executed if used with a context manager.
    """
    with TestClient(app) as local_client:
        response = local_client.get("/feature-importance")
        # Could be 200 if model loads, or 503 if not. We'll simply check it doesn't 500.
        assert response.status_code in [200, 400, 503]


def test_predict_endpoint_success():
    """Test the /predict endpoint with a valid payload."""
    payload = [
        {
            "trip_distance": 2.5,
            "total_amount": 15.0,
            "passenger_count": 1,
            "ratecode_id": 1,
            "airport_fee": 0.0,
            "congestion_surcharge": 2.5,
            "tolls_amount": 0.0,
            "hour": 12,
            "day": 15,
            "month": 1,
        }
    ]

    with TestClient(app) as local_client:
        response = local_client.post("/predict", json=payload)
        # If model is loaded, 200. Else 503.
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            assert len(data) == 1
            assert "predicted_tip" in data[0]
            assert "model_version" in data[0]
        else:
            assert response.status_code == 503
